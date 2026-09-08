#import "LiveMeshStreamer.h"
#import <simd/simd.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreImage/CoreImage.h>
#import <UIKit/UIKit.h>
#import <sys/socket.h>
#import <netinet/in.h>
#import <netinet/tcp.h>
#import <arpa/inet.h>
#import <netdb.h>
#import <unistd.h>
#import <errno.h>
#import <fcntl.h>
#import <sys/select.h>
#import <CoreMotion/CoreMotion.h>
#import <stdatomic.h>

static const uint32_t kProtocolMagic = 0x44544c53;  // "DTLS"
static const uint8_t kProtocolVersion = 1;
static const uint8_t kMsgUpsertAnchor = 1;
static const uint8_t kMsgRemoveAnchor = 2;
static const uint8_t kMsgKeyframeImage = 4;
static const uint8_t kMsgFinalize = 5;
static const uint8_t kMsgKeyframeRGBD = 6;
static const uint8_t kMsgPhonePose = 7;
// Host -> phone control messages (the only inbound types we read).
static const uint8_t kMsgControlStartScan = 8;
static const uint8_t kMsgControlStopScan = 9;
// Phone -> host "reset position": re-anchor the twin alignment to the phone's
// current (mounted) pose. Header-only, like kMsgFinalize.
static const uint8_t kMsgRealign = 10;
static const uint8_t kMsgImu = 12;  // 6 float32 body: accel m/s^2, gyro rad/s
static const uint8_t kMsgControlEnableImu = 13;
static const uint8_t kMsgControlDisableImu = 14;
// Flag bits (wire contract in src/digital_twin/live_scan.py):
//   kFlagHasNormals: trailing per-vertex normal block (smooth shading).
//   kFlagTextured: anchor is textured from a keyframe image (trailing image id
//                  + per-vertex UVs); colors are still sent as a fallback.
static const uint16_t kFlagHasNormals = 0x0001;
static const uint16_t kFlagTextured = 0x0002;

// "Phone set down" guard. As the phone is lowered onto a desk (or covered) the
// camera fills with a very near surface; projecting those frames repaints and
// re-textures nearby anchors with garbage and ruins a finished scan. If more
// than kNearSurfaceFraction of the LiDAR depth map is closer than
// kNearSurfaceMeters, the frame is treated as unusable and skipped entirely.
static const float kNearSurfaceMeters = 0.30f;
static const float kNearSurfaceFraction = 0.6f;
// Downsample keyframes to ~720p before JPEG encoding.
static const CGFloat kKeyframeTargetHeight = 720.0;

// Posed-keyframe capture cadence for the offline bake: capture at most every
// kKeyframeMinInterval seconds, and only once the camera has moved at least
// kKeyframeMinTranslation meters OR rotated past kKeyframeMinRotationDot (the
// cosine of ~5.7 deg between successive camera forward axes). The rotation gate
// is what drives capture for an in-place robot 360 (no translation), so a
// tighter cosine -> denser keyframes -> higher textured-face coverage in the bake.
static const NSTimeInterval kKeyframeMinInterval = 0.5;
static const float kKeyframeMinTranslation = 0.25f;
static const float kKeyframeMinRotationDot = 0.995f;
// Continuous RGB-D includes a JPEG, depth map, and confidence map. 30 fps floods
// the phone send queue and host mapper for ARKit/nvblox operation, making Stop
// Scan wait behind stale frames. Keep the wire rate modest; cuVSLAM-specific
// builds can raise this again when the host can consume every frame.
static const NSTimeInterval kRgbdMinInterval = 1.0 / 10.0;
// Rover-localization pose stream cadence (MSG_PHONE_POSE). RGB-D frames carry
// their own camera pose, so the host only needs this stream for initial/re-align.
static const NSTimeInterval kPoseMinInterval = 0.2;
// Connect timeout. A blocking connect() to an unreachable IP hangs for the OS
// SYN timeout (tens of seconds) and, since connects run on the serial send
// queue, stalls every later Connect tap behind it. Fail fast instead.
static const int kConnectTimeoutSeconds = 3;

@implementation LiveMeshStreamer {
    int _sockfd;
    dispatch_queue_t _sendQueue;
    NSMutableDictionary<NSUUID *, NSNumber *> *_lastSentByAnchor;
    NSMutableDictionary<NSUUID *, NSData *> *_colorsByAnchor;
    CIContext *_ciContext;
    // Posed-keyframe capture state (see considerKeyframeForFrame:).
    simd_float4x4 _lastKeyframeCameraTransform;
    NSTimeInterval _lastKeyframeTime;
    NSUInteger _keyframeCount;
    // Per-frame "is this frame usable" cache, so the depth-map scan runs once per
    // camera frame rather than once per anchor.
    NSTimeInterval _qualityCheckedTimestamp;
    BOOL _qualityCheckedUsable;
    // Continuous rover-localization pose stream (see sendPhonePoseForFrame:).
    uint32_t _poseSequence;
    NSTimeInterval _lastPoseTime;
    // Bumped on every (re)connect so a stale inbound-read loop for an old socket
    // exits instead of reading the new one.
    uint64_t _connectGeneration;
    // CoreMotion IMU for cuVSLAM inertial odometry (continuousRGBD mode).
    CMMotionManager *_motion;
    NSOperationQueue *_imuQueue;
    // RGB-D frames in flight on the (bandwidth-limited) serial send queue. The
    // capture gate skips new frames while this is >0 so we transmit the freshest
    // frame instead of building a backlog -- bounds the twin's lag behind the
    // camera regardless of link speed (see considerKeyframeForFrame:).
    atomic_int _rgbdInFlight;
}

- (instancetype)init {
    if ((self = [super init])) {
        _sockfd = -1;
        // Scene reconstruction can update many anchors per AR frame. Keep the
        // stream deliberately modest so coloring/encoding does not thermal-crash
        // the app or starve ARKit's tracking queue.
        _minimumAnchorInterval = 0.35;
        _sendQueue = dispatch_queue_create("live.scan.send", DISPATCH_QUEUE_SERIAL);
        _lastSentByAnchor = [NSMutableDictionary dictionary];
        _colorsByAnchor = [NSMutableDictionary dictionary];
        _ciContext = [CIContext contextWithOptions:nil];
        _lastKeyframeTime = -1.0;
        _keyframeCount = 0;
        _qualityCheckedTimestamp = -1.0;
        _qualityCheckedUsable = YES;
        _streamingEnabled = YES;
        _imuStreamingEnabled = NO;
        _poseSequence = 0;
        _lastPoseTime = -1.0;
        _motion = [[CMMotionManager alloc] init];
        _imuQueue = [[NSOperationQueue alloc] init];
        _imuQueue.maxConcurrentOperationCount = 1;
    }
    return self;
}

- (void)setStreamingEnabled:(BOOL)enabled {
    _streamingEnabled = enabled;
    if (enabled && self.imuStreamingEnabled) {
        [self startImuUpdates];
    } else {
        [self stopImuUpdates];
    }
}

- (void)setImuStreamingEnabled:(BOOL)enabled {
    _imuStreamingEnabled = enabled;
    if (enabled && self.streamingEnabled) {
        [self startImuUpdates];
    } else {
        [self stopImuUpdates];
    }
}

- (void)startImuUpdates {
    if (!self.continuousRGBD || _motion == nil || _motion.isDeviceMotionActive) {
        return;
    }
    if (!_motion.isDeviceMotionAvailable) {
        return;
    }
    _motion.deviceMotionUpdateInterval = 1.0 / 100.0;  // 100 Hz
    __weak typeof(self) weakSelf = self;
    [_motion startDeviceMotionUpdatesToQueue:_imuQueue
                                 withHandler:^(CMDeviceMotion *m, NSError *error) {
        if (m == nil || error != nil) {
            return;
        }
        // cuVSLAM wants specific force (m/s^2, incl. gravity) + angular velocity
        // (rad/s) in the IMU frame. Apple splits raw accel into gravity +
        // userAcceleration (both in g); recombine and scale. Sign/frame may need
        // tuning on the host (rig_from_imu) -- watch the gravity-down check.
        const double g0 = 9.80665;
        simd_float3 accel = simd_make_float3((m.gravity.x + m.userAcceleration.x) * g0,
                                             (m.gravity.y + m.userAcceleration.y) * g0,
                                             (m.gravity.z + m.userAcceleration.z) * g0);
        simd_float3 gyro = simd_make_float3(m.rotationRate.x, m.rotationRate.y, m.rotationRate.z);
        [weakSelf sendImuWithAccel:accel gyro:gyro timestamp:m.timestamp];
    }];
}

- (void)stopImuUpdates {
    if (_motion.isDeviceMotionActive) {
        [_motion stopDeviceMotionUpdates];
    }
}

// MSG_IMU: header + 6 float32 body (accel x/y/z, gyro x/y/z). Sent at ~100 Hz.
- (void)sendImuWithAccel:(simd_float3)accel gyro:(simd_float3)gyro timestamp:(double)timestamp {
    if (!_connected) {
        return;
    }
    NSMutableData *payload = [NSMutableData data];
    [self appendHeaderTo:payload
              messageType:kMsgImu
                    flags:0
                 anchorId:[[NSUUID alloc] initWithUUIDBytes:(const unsigned char[16]){0}]
                timestamp:timestamp
                transform:matrix_identity_float4x4
              vertexCount:0
                faceCount:0];
    float body[6] = {accel.x, accel.y, accel.z, gyro.x, gyro.y, gyro.z};
    [payload appendBytes:body length:sizeof(body)];
    NSData *message = [self frameMessage:payload];
    dispatch_async(_sendQueue, ^{
        if (!self.streamingEnabled || !self.imuStreamingEnabled) {
            return;  // stale IMU captured before Stop Scan
        }
        [self _sendAllLocked:message];
    });
}

- (void)dealloc {
    [self disconnect];
}

#pragma mark - Connection

- (void)connectToHost:(NSString *)host port:(uint16_t)port {
    dispatch_async(_sendQueue, ^{
        [self _closeLocked];

        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            return;
        }

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);

        if (inet_pton(AF_INET, host.UTF8String, &addr.sin_addr) != 1) {
            // Not a dotted-quad; resolve as a hostname.
            struct hostent *he = gethostbyname(host.UTF8String);
            if (he == NULL || he->h_addr_list[0] == NULL) {
                close(fd);
                return;
            }
            memcpy(&addr.sin_addr, he->h_addr_list[0], (size_t)he->h_length);
        }

        // Non-blocking connect with a bounded wait: a wrong/unreachable IP fails
        // in kConnectTimeoutSeconds instead of hanging the send queue for the
        // full OS SYN timeout (which made later Connect taps appear stuck).
        int flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        int rc = connect(fd, (struct sockaddr *)&addr, sizeof(addr));
        if (rc != 0) {
            if (errno != EINPROGRESS) {
                close(fd);
                return;
            }
            fd_set wset;
            FD_ZERO(&wset);
            FD_SET(fd, &wset);
            struct timeval tv = { .tv_sec = kConnectTimeoutSeconds, .tv_usec = 0 };
            int sel = select(fd + 1, NULL, &wset, NULL, &tv);
            if (sel <= 0) {
                close(fd);  // 0 = timed out, <0 = error
                return;
            }
            int soerr = 0;
            socklen_t errlen = sizeof(soerr);
            if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &errlen) != 0 || soerr != 0) {
                close(fd);
                return;
            }
        }
        fcntl(fd, F_SETFL, flags);  // restore blocking for the send/recv paths

        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
        // Without this, iOS may terminate the process with SIGPIPE if Isaac is
        // closed while a send is in flight.
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
        _sockfd = fd;
        _connected = YES;
        [_lastSentByAnchor removeAllObjects];
        [_colorsByAnchor removeAllObjects];
        _lastKeyframeTime = -1.0;
        _keyframeCount = 0;
        // Start reading host->phone control frames on this connection. Run it on
        // a separate queue: _sendQueue is serial and must stay free for sends,
        // and recv() blocks. Capture (fd, generation) so the loop self-exits when
        // the socket is closed/replaced.
        uint64_t generation = ++_connectGeneration;
        dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
            [self _runControlReadLoopForFD:fd generation:generation];
        });
    });
}

- (void)disconnect {
    dispatch_sync(_sendQueue, ^{
        [self _closeLocked];
    });
}

- (void)_closeLocked {
    [self stopImuUpdates];
    if (_sockfd >= 0) {
        close(_sockfd);
        _sockfd = -1;
    }
    _connected = NO;
    _connectGeneration++;  // invalidate any in-flight control read loop
}

// Blocking full-buffer write; runs on _sendQueue. Marks disconnected on error.
- (BOOL)_sendAllLocked:(NSData *)data {
    if (_sockfd < 0) {
        return NO;
    }
    const uint8_t *bytes = data.bytes;
    size_t remaining = data.length;
    while (remaining > 0) {
        ssize_t n = send(_sockfd, bytes, remaining, 0);
        if (n <= 0) {
            if (errno == EINTR) {
                continue;
            }
            [self _closeLocked];
            return NO;
        }
        bytes += n;
        remaining -= (size_t)n;
    }
    return YES;
}

#pragma mark - Receiving (host -> phone control)

// Blocking read of exactly `length` bytes from `fd`. Returns NO on peer close,
// error, or if this connection has been superseded (generation changed).
- (BOOL)_readExactly:(uint8_t *)buf length:(uint32_t)length fd:(int)fd generation:(uint64_t)generation {
    size_t got = 0;
    while (got < length) {
        if (_connectGeneration != generation) {
            return NO;
        }
        ssize_t n = recv(fd, buf + got, length - got, 0);
        if (n > 0) {
            got += (size_t)n;
            continue;
        }
        if (n < 0 && errno == EINTR) {
            continue;
        }
        return NO;  // 0 = peer closed, <0 = error
    }
    return YES;
}

// Reads length-prefixed control frames host->phone and forwards START/STOP to
// onRemoteControl on the main queue. Runs off _sendQueue for the life of one
// connection; exits when the socket closes or is replaced.
- (void)_runControlReadLoopForFD:(int)fd generation:(uint64_t)generation {
    while (_connectGeneration == generation) {
        uint8_t lenbuf[4];
        if (![self _readExactly:lenbuf length:4 fd:fd generation:generation]) {
            return;
        }
        uint32_t payloadSize = (uint32_t)lenbuf[0] | ((uint32_t)lenbuf[1] << 8)
                             | ((uint32_t)lenbuf[2] << 16) | ((uint32_t)lenbuf[3] << 24);
        // Control frames are header-only (~104 bytes). Guard against junk.
        if (payloadSize < 6 || payloadSize > (1u << 20)) {
            return;
        }
        uint8_t header[6];
        if (![self _readExactly:header length:6 fd:fd generation:generation]) {
            return;
        }
        // Drain any payload bytes past the 6-byte prefix we parse.
        uint32_t remaining = payloadSize - 6;
        uint8_t scratch[512];
        while (remaining > 0) {
            uint32_t chunk = remaining < sizeof(scratch) ? remaining : (uint32_t)sizeof(scratch);
            if (![self _readExactly:scratch length:chunk fd:fd generation:generation]) {
                return;
            }
            remaining -= chunk;
        }
        uint32_t magic = (uint32_t)header[0] | ((uint32_t)header[1] << 8)
                       | ((uint32_t)header[2] << 16) | ((uint32_t)header[3] << 24);
        uint8_t version = header[4];
        uint8_t messageType = header[5];
        if (magic != kProtocolMagic || version != kProtocolVersion) {
            continue;  // ignore frames we don't understand
        }
        if (messageType == kMsgControlEnableImu) {
            self.imuStreamingEnabled = YES;
            continue;
        }
        if (messageType == kMsgControlDisableImu) {
            self.imuStreamingEnabled = NO;
            continue;
        }
        if (messageType == kMsgControlStartScan || messageType == kMsgControlStopScan) {
            void (^callback)(uint8_t) = self.onRemoteControl;
            if (callback != nil) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    callback(messageType);
                });
            }
        }
    }
}

#pragma mark - Sending

- (void)handleMeshAnchor:(ARMeshAnchor *)anchor frame:(ARFrame *)frame {
    if (!_connected || !_streamingEnabled || frame == nil) {
        return;
    }
    // Continuous RGB-D (nvblox) mode: the host builds geometry from depth and
    // discards mesh anchors, so don't encode/colour/stream them -- it's pure
    // wasted phone CPU and link bandwidth (~36% of the wire, measured). Scene
    // reconstruction is normally off in this mode too (see ScannerViewController);
    // this guards the path in case an anchor still arrives.
    if (_continuousRGBD) {
        return;
    }
    // Skip the whole camera frame when the phone is being set down against a
    // surface (computed once per frame, shared across all its anchors).
    if (frame.timestamp != _qualityCheckedTimestamp) {
        _qualityCheckedTimestamp = frame.timestamp;
        _qualityCheckedUsable = [self frameUsableForColoring:frame];
    }
    if (!_qualityCheckedUsable) {
        return;
    }
    NSTimeInterval now = CACurrentMediaTime();
    NSNumber *last = _lastSentByAnchor[anchor.identifier];
    if (last != nil && now - last.doubleValue < _minimumAnchorInterval) {
        return;
    }
    _lastSentByAnchor[anchor.identifier] = @(now);

    NSData *message = [self encodeUpsert:anchor frame:frame];
    if (message == nil) {
        return;
    }
    dispatch_async(_sendQueue, ^{
        if (!self.streamingEnabled) {
            return;  // stale mesh update captured before Stop Scan
        }
        if ([self _sendAllLocked:message]) {
            _framesSent += 1;
        }
    });
}

// Reject frames captured with the camera right up against a surface — the
// signature of the phone being lowered onto a desk at the end of a scan. Such
// frames, if projected, repaint and re-texture nearby anchors with a near/dark
// close-up. Returns NO when most of the LiDAR depth map is nearer than
// kNearSurfaceMeters. When there is no depth map, assume usable.
- (BOOL)frameUsableForColoring:(ARFrame *)frame {
    ARDepthData *depthData = frame.smoothedSceneDepth ?: frame.sceneDepth;
    CVPixelBufferRef depthMap = depthData.depthMap;
    if (depthMap == NULL) {
        return YES;
    }
    CVPixelBufferLockBaseAddress(depthMap, kCVPixelBufferLock_ReadOnly);
    size_t width = CVPixelBufferGetWidth(depthMap);
    size_t height = CVPixelBufferGetHeight(depthMap);
    const uint8_t *base = (const uint8_t *)CVPixelBufferGetBaseAddress(depthMap);
    size_t rowBytes = CVPixelBufferGetBytesPerRow(depthMap);
    NSUInteger nearCount = 0;
    NSUInteger total = 0;
    // Sparse 8x8 stride is plenty to characterize the whole frame cheaply.
    for (size_t y = 0; y < height; y += 8) {
        const float *row = (const float *)(base + y * rowBytes);
        for (size_t x = 0; x < width; x += 8) {
            float d = row[x];
            if (d > 0.0f) {
                total++;
                if (d < kNearSurfaceMeters) {
                    nearCount++;
                }
            }
        }
    }
    CVPixelBufferUnlockBaseAddress(depthMap, kCVPixelBufferLock_ReadOnly);
    if (total > 0 && (float)nearCount / (float)total > kNearSurfaceFraction) {
        return NO;
    }
    return YES;
}

- (BOOL)frameHasNormalTracking:(ARFrame *)frame {
    return frame.camera.trackingState == ARTrackingStateNormal;
}

- (void)removeAnchorWithId:(NSUUID *)anchorId {
    [_colorsByAnchor removeObjectForKey:anchorId];
    NSMutableData *payload = [NSMutableData data];
    [self appendHeaderTo:payload
              messageType:kMsgRemoveAnchor
                    flags:0
                 anchorId:anchorId
                timestamp:CACurrentMediaTime()
                transform:matrix_identity_float4x4
              vertexCount:0
                faceCount:0];
    NSData *message = [self frameMessage:payload];
    dispatch_async(_sendQueue, ^{
        [self _sendAllLocked:message];
    });
}

#pragma mark - Encoding

- (nullable NSData *)encodeUpsert:(ARMeshAnchor *)meshAnchor frame:(ARFrame *)frame {
    ARMeshGeometry *geometry = meshAnchor.geometry;
    ARGeometrySource *vertexSource = geometry.vertices;
    ARGeometryElement *faceElement = geometry.faces;
    NSUInteger vertexCount = vertexSource.count;
    NSUInteger faceCount = faceElement.count;
    if (vertexCount == 0 || faceCount == 0) {
        return nil;
    }

    // ARKit supplies a per-vertex normal source (same count/layout as vertices)
    // in the anchor's local frame. Stream it so the host can smooth-shade.
    ARGeometrySource *normalSource = geometry.normals;
    BOOL hasNormals = (normalSource != nil && normalSource.count == vertexCount);

    const char *vBase = (const char *)vertexSource.buffer.contents;
    NSUInteger vOffset = vertexSource.offset;
    NSUInteger vStride = vertexSource.stride;
    const char *nBase = hasNormals ? (const char *)normalSource.buffer.contents : NULL;
    NSUInteger nOffset = hasNormals ? normalSource.offset : 0;
    NSUInteger nStride = hasNormals ? normalSource.stride : 0;

    // ---- vertex block (local ARMeshAnchor frame) ----
    NSMutableData *vertexData = [NSMutableData dataWithCapacity:vertexCount * 12];
    for (NSUInteger i = 0; i < vertexCount; i++) {
        const float *v = (const float *)(vBase + vOffset + vStride * i);
        float xyz[3] = {v[0], v[1], v[2]};
        [vertexData appendBytes:xyz length:sizeof(xyz)];
    }

    // ---- face block (uint32 triangle indices) ----
    NSMutableData *faceData = [NSMutableData dataWithCapacity:faceCount * 12];
    const char *fBase = (const char *)faceElement.buffer.contents;
    NSUInteger bytesPerIndex = faceElement.bytesPerIndex;
    NSUInteger idxPerPrim = faceElement.indexCountPerPrimitive;  // 3 for triangles
    for (NSUInteger f = 0; f < faceCount; f++) {
        const char *tri = fBase + f * idxPerPrim * bytesPerIndex;
        for (NSUInteger k = 0; k < 3; k++) {
            uint32_t index = (bytesPerIndex == 2) ? (uint32_t)(((const uint16_t *)tri)[k])
                                                  : ((const uint32_t *)tri)[k];
            [faceData appendBytes:&index length:sizeof(index)];
        }
    }

    // ---- color block + coverage ----
    // Persist per-vertex color and only overwrite a vertex when the current
    // frame yields a VALID sample (front-facing, in-frame, not occluded), so
    // off-screen geometry keeps its color instead of graying out. The count of
    // valid samples is the anchor's "coverage" in this frame, used to decide
    // whether to (re)capture a texture keyframe below.
    NSMutableData *colors = [NSMutableData dataWithLength:vertexCount * 3];
    uint8_t *colorBytes = (uint8_t *)colors.mutableBytes;
    NSData *cachedColors = _colorsByAnchor[meshAnchor.identifier];
    NSUInteger seededBytes = 0;
    if (cachedColors != nil) {
        seededBytes = MIN(cachedColors.length, vertexCount * 3);
        memcpy(colorBytes, cachedColors.bytes, seededBytes);
    }
    for (NSUInteger b = seededBytes; b < vertexCount * 3; b++) {
        colorBytes[b] = 200;  // neutral fallback for not-yet-seen vertices
    }

    simd_float4x4 cameraTransform = frame.camera.transform;
    simd_float4x4 invCamera = simd_inverse(cameraTransform);
    matrix_float3x3 intrinsics = frame.camera.intrinsics;
    simd_float3 cameraPosition = simd_make_float3(cameraTransform.columns[3].x,
                                                  cameraTransform.columns[3].y,
                                                  cameraTransform.columns[3].z);
    simd_float4x4 anchorT = meshAnchor.transform;
    matrix_float3x3 anchorRotation = simd_matrix(
        simd_make_float3(anchorT.columns[0].x, anchorT.columns[0].y, anchorT.columns[0].z),
        simd_make_float3(anchorT.columns[1].x, anchorT.columns[1].y, anchorT.columns[1].z),
        simd_make_float3(anchorT.columns[2].x, anchorT.columns[2].y, anchorT.columns[2].z));

    ARDepthData *depthData = frame.smoothedSceneDepth ?: frame.sceneDepth;
    CVPixelBufferRef depthMap = depthData.depthMap;
    CVPixelBufferRef confidenceMap = depthData.confidenceMap;

    CVPixelBufferRef image = frame.capturedImage;
    float imageWidth = (float)CVPixelBufferGetWidthOfPlane(image, 0);
    float imageHeight = (float)CVPixelBufferGetHeightOfPlane(image, 0);
    (void)imageWidth;
    (void)imageHeight;

    CVPixelBufferLockBaseAddress(image, kCVPixelBufferLock_ReadOnly);
    if (depthMap != NULL) {
        CVPixelBufferLockBaseAddress(depthMap, kCVPixelBufferLock_ReadOnly);
    }
    if (confidenceMap != NULL) {
        CVPixelBufferLockBaseAddress(confidenceMap, kCVPixelBufferLock_ReadOnly);
    }
    NSUInteger validCount = 0;
    for (NSUInteger i = 0; i < vertexCount; i++) {
        const float *v = (const float *)(vBase + vOffset + vStride * i);
        simd_float4 worldPoint = simd_mul(anchorT, simd_make_float4(v[0], v[1], v[2], 1.0f));

        // Back-face rejection: skip surfaces facing away from the camera so the
        // image is never painted "through" geometry onto its far side.
        if (hasNormals) {
            const float *n = (const float *)(nBase + nOffset + nStride * i);
            simd_float3 worldNormal = simd_mul(anchorRotation, simd_make_float3(n[0], n[1], n[2]));
            simd_float3 viewRay = simd_make_float3(worldPoint.x - cameraPosition.x,
                                                   worldPoint.y - cameraPosition.y,
                                                   worldPoint.z - cameraPosition.z);
            if (simd_dot(worldNormal, viewRay) >= 0.0f) {
                continue;
            }
        }

        uint8_t rgb[3];
        BOOL valid = [self sampleColor:rgb
                            worldPoint:worldPoint
                             invCamera:invCamera
                            intrinsics:intrinsics
                                 image:image
                              depthMap:depthMap
                            confidence:confidenceMap];
        if (valid) {
            colorBytes[i * 3 + 0] = rgb[0];
            colorBytes[i * 3 + 1] = rgb[1];
            colorBytes[i * 3 + 2] = rgb[2];
            validCount++;
        }
    }
    if (confidenceMap != NULL) {
        CVPixelBufferUnlockBaseAddress(confidenceMap, kCVPixelBufferLock_ReadOnly);
    }
    if (depthMap != NULL) {
        CVPixelBufferUnlockBaseAddress(depthMap, kCVPixelBufferLock_ReadOnly);
    }
    CVPixelBufferUnlockBaseAddress(image, kCVPixelBufferLock_ReadOnly);
    _colorsByAnchor[meshAnchor.identifier] = colors;

    // Live anchors are NOT projectively textured: a frozen single keyframe has no
    // occlusion test, so anything in front of a surface (a person before a wall)
    // bleeds onto it. Live shows occlusion-tested vertex color only; high-quality
    // texture baking belongs to the separate research workflow.
    (void)validCount;
    BOOL textured = NO;

    // ---- assemble: header, vertices, faces, colors, [normals] ----
    uint16_t flags = (uint16_t)((hasNormals ? kFlagHasNormals : 0) | (textured ? kFlagTextured : 0));
    NSMutableData *payload = [NSMutableData data];
    [self appendHeaderTo:payload
              messageType:kMsgUpsertAnchor
                    flags:flags
                 anchorId:meshAnchor.identifier
                timestamp:frame.timestamp
                transform:meshAnchor.transform
              vertexCount:(uint32_t)vertexCount
                faceCount:(uint32_t)faceCount];
    [payload appendData:vertexData];
    [payload appendData:faceData];
    [payload appendData:colors];
    if (hasNormals) {
        for (NSUInteger i = 0; i < vertexCount; i++) {
            const float *n = (const float *)(nBase + nOffset + nStride * i);
            float xyz[3] = {n[0], n[1], n[2]};
            [payload appendBytes:xyz length:sizeof(xyz)];
        }
    }
    return [self frameMessage:payload];
}

// Capture a posed keyframe for the offline bake when the camera has moved
// enough since the last one. Sends MSG_KEYFRAME_IMAGE carrying the camera pose
// and intrinsics (scaled to the downsampled JPEG resolution). Runs on the ARKit
// delegate thread; the JPEG encode is cheap (~2/s) and the send is dispatched.
- (void)considerKeyframeForFrame:(ARFrame *)frame {
    if (!_connected || !_streamingEnabled || frame == nil) {
        return;
    }
    // Reuse the per-frame usability cache (skips near/dark frames).
    if (frame.timestamp != _qualityCheckedTimestamp) {
        _qualityCheckedTimestamp = frame.timestamp;
        _qualityCheckedUsable = [self frameUsableForColoring:frame];
    }
    // cuVSLAM wants a steady stream including during motion, so it does its own
    // tracking -- skip the bake-oriented coloring/normal-tracking gates in that
    // mode (ARKit's pose is only used to anchor gravity on the first frame).
    if (!_continuousRGBD) {
        if (!_qualityCheckedUsable) {
            return;
        }
        // Bad/limited ARKit tracking is especially damaging to the offline texture
        // bake: the mesh may still look plausible after ARKit refines anchors, but
        // a stale keyframe pose projects pixels onto the wrong surface.
        if (![self frameHasNormalTracking:frame]) {
            return;
        }
    }

    simd_float4x4 cam = frame.camera.transform;
    NSTimeInterval now = CACurrentMediaTime();
    if (_continuousRGBD) {
        // Continuous cadence only -- no motion gate (cuVSLAM needs every frame).
        if (_keyframeCount > 0 && now - _lastKeyframeTime < kRgbdMinInterval) {
            return;
        }
        // Bandwidth-adaptive drop: if the previous RGB-D frame hasn't drained off
        // the serial send queue yet, skip this one (before the costly JPEG/depth
        // encode) so we always send the FRESHEST frame rather than queue a
        // backlog that lands the twin seconds behind the camera. The effective
        // rate self-adjusts to min(kRgbdMinInterval, link capacity).
        if (atomic_load(&_rgbdInFlight) > 0) {
            return;
        }
    } else if (_keyframeCount > 0) {
        if (now - _lastKeyframeTime < kKeyframeMinInterval) {
            return;
        }
        simd_float3 dp = simd_make_float3(cam.columns[3].x - _lastKeyframeCameraTransform.columns[3].x,
                                          cam.columns[3].y - _lastKeyframeCameraTransform.columns[3].y,
                                          cam.columns[3].z - _lastKeyframeCameraTransform.columns[3].z);
        float dist = simd_length(dp);
        // Camera forward axis is -Z; compare successive forward directions.
        simd_float3 f0 = simd_make_float3(_lastKeyframeCameraTransform.columns[2].x,
                                          _lastKeyframeCameraTransform.columns[2].y,
                                          _lastKeyframeCameraTransform.columns[2].z);
        simd_float3 f1 = simd_make_float3(cam.columns[2].x, cam.columns[2].y, cam.columns[2].z);
        float forwardDot = simd_dot(simd_normalize(f0), simd_normalize(f1));
        if (dist < kKeyframeMinTranslation && forwardDot > kKeyframeMinRotationDot) {
            return;  // not moved/rotated enough to add a new view
        }
    }

    uint32_t width = 0;
    uint32_t height = 0;
    NSData *jpeg = [self jpegFromPixelBuffer:frame.capturedImage outWidth:&width outHeight:&height];
    if (jpeg == nil) {
        return;
    }
    // Scale full-res intrinsics to the downsampled JPEG pixel space.
    matrix_float3x3 K = frame.camera.intrinsics;
    float fullHeight = (float)CVPixelBufferGetHeightOfPlane(frame.capturedImage, 0);
    float scale = (height > 0 && fullHeight > 0.0f) ? (float)height / fullHeight : 1.0f;
    float fx = K.columns[0][0] * scale;
    float fy = K.columns[1][1] * scale;
    float cx = K.columns[2][0] * scale;
    float cy = K.columns[2][1] * scale;

    ARDepthData *depthData = frame.smoothedSceneDepth ?: frame.sceneDepth;
    uint32_t depthWidth = 0;
    uint32_t depthHeight = 0;
    NSData *depthBytes = nil;
    NSData *confidenceBytes = nil;
    if (depthData.depthMap != NULL) {
        depthBytes = [self contiguousFloatDepthDataFromPixelBuffer:depthData.depthMap
                                                          outWidth:&depthWidth
                                                         outHeight:&depthHeight];
        if (depthBytes != nil && depthData.confidenceMap != NULL) {
            uint32_t confidenceWidth = 0;
            uint32_t confidenceHeight = 0;
            confidenceBytes = [self contiguousUInt8DataFromPixelBuffer:depthData.confidenceMap
                                                               outWidth:&confidenceWidth
                                                              outHeight:&confidenceHeight];
            if (confidenceWidth != depthWidth || confidenceHeight != depthHeight) {
                confidenceBytes = nil;
            }
        }
    }

    // cuVSLAM needs RGB-D: drop frames without depth instead of sending RGB-only.
    if (_continuousRGBD && depthBytes == nil) {
        return;
    }

    NSUUID *imageId = [NSUUID UUID];
    NSData *message = [self encodeKeyframeImage:imageId
                                cameraTransform:cam
                                             fx:fx fy:fy cx:cx cy:cy
                                          width:width height:height
                                           jpeg:jpeg
                                     depthWidth:depthWidth
                                    depthHeight:depthHeight
                                     depthBytes:depthBytes
                                confidenceBytes:confidenceBytes
                                      timestamp:frame.timestamp];
    atomic_fetch_add(&_rgbdInFlight, 1);  // cleared in the block (the in-flight gate above)
    dispatch_async(_sendQueue, ^{
        if (self.streamingEnabled && [self _sendAllLocked:message]) {
            _keyframesSent += 1;
        }
        atomic_fetch_sub(&_rgbdInFlight, 1);
    });
    _lastKeyframeCameraTransform = cam;
    _lastKeyframeTime = now;
    _keyframeCount += 1;
}

// Tell the host to bake the accumulated mesh + keyframes into a textured mesh.
- (void)sendFinalize {
    NSMutableData *payload = [NSMutableData data];
    [self appendHeaderTo:payload
              messageType:kMsgFinalize
                    flags:0
                 anchorId:[[NSUUID alloc] initWithUUIDBytes:(const unsigned char[16]){0}]
                timestamp:CACurrentMediaTime()
                transform:matrix_identity_float4x4
              vertexCount:0
                faceCount:0];
    NSData *message = [self frameMessage:payload];
    dispatch_async(_sendQueue, ^{
        [self _sendAllLocked:message];
    });
}

// Tell the host to re-anchor the twin alignment to the phone's current pose
// (Reset Position). Header-only, sent whether or not streaming is enabled so the
// operator can zero the origin before starting a scan.
- (void)sendRealign {
    NSMutableData *payload = [NSMutableData data];
    [self appendHeaderTo:payload
              messageType:kMsgRealign
                    flags:0
                 anchorId:[[NSUUID alloc] initWithUUIDBytes:(const unsigned char[16]){0}]
                timestamp:CACurrentMediaTime()
                transform:matrix_identity_float4x4
              vertexCount:0
                faceCount:0];
    NSData *message = [self frameMessage:payload];
    dispatch_async(_sendQueue, ^{
        [self _sendAllLocked:message];
    });
}

// Map ARCamera.trackingState to the host's TRACKING_* codes
// (notAvailable=0, limited=1, normal=2).
static uint32_t TrackingStateCode(ARTrackingState state) {
    switch (state) {
        case ARTrackingStateNormal: return 2;
        case ARTrackingStateLimited: return 1;
        case ARTrackingStateNotAvailable: return 0;
    }
    return 0;
}

- (void)sendPhonePoseForFrame:(ARFrame *)frame {
    // Gated only on the connection (not streamingEnabled): localization must
    // continue even when mesh streaming is paused.
    if (!_connected || frame == nil) {
        return;
    }
    NSTimeInterval now = CACurrentMediaTime();
    if (_lastPoseTime >= 0.0 && now - _lastPoseTime < kPoseMinInterval) {
        return;
    }
    _lastPoseTime = now;

    NSMutableData *payload = [NSMutableData data];
    [self appendHeaderTo:payload
              messageType:kMsgPhonePose
                    flags:0
                 anchorId:[[NSUUID alloc] initWithUUIDBytes:(const unsigned char[16]){0}]
                timestamp:frame.timestamp
                transform:frame.camera.transform
              vertexCount:0
                faceCount:0];
    uint32_t trackingState = TrackingStateCode(frame.camera.trackingState);
    uint32_t sequence = _poseSequence++;
    [payload appendBytes:&trackingState length:sizeof(trackingState)];
    [payload appendBytes:&sequence length:sizeof(sequence)];
    NSData *message = [self frameMessage:payload];
    dispatch_async(_sendQueue, ^{
        [self _sendAllLocked:message];
    });
}

// Downsample the captured frame to ~kKeyframeTargetHeight and JPEG-encode it.
// UVs are normalized, so the downsampled resolution does not affect mapping.
- (nullable NSData *)jpegFromPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                outWidth:(uint32_t *)outWidth
                               outHeight:(uint32_t *)outHeight {
    CIImage *ciImage = [CIImage imageWithCVPixelBuffer:pixelBuffer];
    CGFloat fullHeight = ciImage.extent.size.height;
    CGFloat scale = (fullHeight > kKeyframeTargetHeight) ? (kKeyframeTargetHeight / fullHeight) : 1.0;
    CIImage *scaled = (scale < 1.0)
        ? [ciImage imageByApplyingTransform:CGAffineTransformMakeScale(scale, scale)]
        : ciImage;
    CGImageRef cgImage = [_ciContext createCGImage:scaled fromRect:scaled.extent];
    if (cgImage == NULL) {
        return nil;
    }
    UIImage *uiImage = [UIImage imageWithCGImage:cgImage];
    NSData *jpeg = UIImageJPEGRepresentation(uiImage, 0.7);
    if (outWidth != NULL) {
        *outWidth = (uint32_t)CGImageGetWidth(cgImage);
    }
    if (outHeight != NULL) {
        *outHeight = (uint32_t)CGImageGetHeight(cgImage);
    }
    CGImageRelease(cgImage);
    return jpeg;
}

- (nullable NSData *)contiguousFloatDepthDataFromPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                                   outWidth:(uint32_t *)outWidth
                                                  outHeight:(uint32_t *)outHeight {
    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    size_t width = CVPixelBufferGetWidth(pixelBuffer);
    size_t height = CVPixelBufferGetHeight(pixelBuffer);
    const uint8_t *base = (const uint8_t *)CVPixelBufferGetBaseAddress(pixelBuffer);
    size_t rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer);
    NSMutableData *data = [NSMutableData dataWithLength:width * height * sizeof(float)];
    uint8_t *dst = (uint8_t *)data.mutableBytes;
    for (size_t y = 0; y < height; y++) {
        memcpy(dst + y * width * sizeof(float), base + y * rowBytes, width * sizeof(float));
    }
    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    if (outWidth != NULL) {
        *outWidth = (uint32_t)width;
    }
    if (outHeight != NULL) {
        *outHeight = (uint32_t)height;
    }
    return data;
}

- (nullable NSData *)contiguousUInt8DataFromPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                               outWidth:(uint32_t *)outWidth
                                              outHeight:(uint32_t *)outHeight {
    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    size_t width = CVPixelBufferGetWidth(pixelBuffer);
    size_t height = CVPixelBufferGetHeight(pixelBuffer);
    const uint8_t *base = (const uint8_t *)CVPixelBufferGetBaseAddress(pixelBuffer);
    size_t rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer);
    NSMutableData *data = [NSMutableData dataWithLength:width * height];
    uint8_t *dst = (uint8_t *)data.mutableBytes;
    for (size_t y = 0; y < height; y++) {
        memcpy(dst + y * width, base + y * rowBytes, width);
    }
    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    if (outWidth != NULL) {
        *outWidth = (uint32_t)width;
    }
    if (outHeight != NULL) {
        *outHeight = (uint32_t)height;
    }
    return data;
}

- (NSData *)encodeKeyframeImage:(NSUUID *)imageId
                cameraTransform:(simd_float4x4)cameraTransform
                             fx:(float)fx fy:(float)fy cx:(float)cx cy:(float)cy
                          width:(uint32_t)width height:(uint32_t)height
                           jpeg:(NSData *)jpeg
                     depthWidth:(uint32_t)depthWidth
                    depthHeight:(uint32_t)depthHeight
                     depthBytes:(nullable NSData *)depthBytes
                confidenceBytes:(nullable NSData *)confidenceBytes
                      timestamp:(double)timestamp {
    NSMutableData *payload = [NSMutableData data];
    // Camera-to-world pose travels in the header transform field; the bake reads
    // it back as the keyframe's camera pose.
    BOOL hasDepth = (depthBytes != nil && depthBytes.length == (NSUInteger)depthWidth * (NSUInteger)depthHeight * sizeof(float));
    [self appendHeaderTo:payload
              messageType:(hasDepth ? kMsgKeyframeRGBD : kMsgKeyframeImage)
                    flags:0
                 anchorId:imageId
                timestamp:timestamp
                transform:cameraTransform
              vertexCount:0
                faceCount:0];
    uint32_t jpegLen = (uint32_t)jpeg.length;
    [payload appendBytes:&fx length:sizeof(fx)];
    [payload appendBytes:&fy length:sizeof(fy)];
    [payload appendBytes:&cx length:sizeof(cx)];
    [payload appendBytes:&cy length:sizeof(cy)];
    [payload appendBytes:&width length:sizeof(width)];
    [payload appendBytes:&height length:sizeof(height)];
    [payload appendBytes:&jpegLen length:sizeof(jpegLen)];
    [payload appendData:jpeg];
    if (hasDepth) {
        uint32_t depthLen = (uint32_t)depthBytes.length;
        uint32_t confidenceLen = (uint32_t)(confidenceBytes != nil ? confidenceBytes.length : 0);
        [payload appendBytes:&depthWidth length:sizeof(depthWidth)];
        [payload appendBytes:&depthHeight length:sizeof(depthHeight)];
        [payload appendBytes:&depthLen length:sizeof(depthLen)];
        [payload appendBytes:&confidenceLen length:sizeof(confidenceLen)];
        [payload appendData:depthBytes];
        if (confidenceBytes != nil) {
            [payload appendData:confidenceBytes];
        }
    }
    return [self frameMessage:payload];
}

// Header layout matches the live nvblox wire contract. iOS arm64 and the x86 host
// are both little-endian, so fields are appended in native byte order.
- (void)appendHeaderTo:(NSMutableData *)payload
           messageType:(uint8_t)messageType
                 flags:(uint16_t)flags
              anchorId:(NSUUID *)anchorId
             timestamp:(double)timestamp
             transform:(simd_float4x4)transform
           vertexCount:(uint32_t)vertexCount
             faceCount:(uint32_t)faceCount {
    uint32_t magic = kProtocolMagic;
    uint8_t version = kProtocolVersion;
    [payload appendBytes:&magic length:sizeof(magic)];
    [payload appendBytes:&version length:sizeof(version)];
    [payload appendBytes:&messageType length:sizeof(messageType)];
    [payload appendBytes:&flags length:sizeof(flags)];

    uuid_t anchorBytes;
    [anchorId getUUIDBytes:anchorBytes];
    [payload appendBytes:anchorBytes length:sizeof(anchorBytes)];

    [payload appendBytes:&timestamp length:sizeof(timestamp)];

    // ARKit's matrix_float4x4 is column-major with translation in columns[3]
    // (column-vector convention). The host applies a row-vector transform with
    // translation in flat indices 12-14, so it needs the TRANSPOSE, which is
    // exactly ARKit's column-major storage emitted in natural order:
    // flat[i*4 + j] = transform.columns[i][j]. (Matches the fixed Swift port.)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float value = transform.columns[i][j];
            [payload appendBytes:&value length:sizeof(value)];
        }
    }

    [payload appendBytes:&vertexCount length:sizeof(vertexCount)];
    [payload appendBytes:&faceCount length:sizeof(faceCount)];
}

- (NSData *)frameMessage:(NSData *)payload {
    uint32_t length = (uint32_t)payload.length;
    NSMutableData *message = [NSMutableData dataWithCapacity:sizeof(length) + payload.length];
    [message appendBytes:&length length:sizeof(length)];
    [message appendData:payload];
    return message;
}

// Returns YES and writes outRGB when the world point projects into the current
// RGB frame (in front of the camera, on-screen, and not occluded by nearer
// geometry per the LiDAR depth map); returns NO otherwise and leaves outRGB
// untouched so the caller can keep the vertex's prior color. depthMap/confidence
// may be NULL (no occlusion test); both are assumed already locked by the caller.
- (BOOL)sampleColor:(uint8_t *)outRGB
         worldPoint:(simd_float4)worldPoint
          invCamera:(simd_float4x4)invCamera
         intrinsics:(matrix_float3x3)intrinsics
              image:(CVPixelBufferRef)image
           depthMap:(CVPixelBufferRef)depthMap
         confidence:(CVPixelBufferRef)confidenceMap {
    simd_float4 cameraPoint = simd_mul(invCamera, worldPoint);
    if (cameraPoint.z >= -0.001f) {
        return NO;
    }

    float x = cameraPoint.x / -cameraPoint.z;
    float y = cameraPoint.y / -cameraPoint.z;
    // intrinsics.columns[col][row]: fx=[0][0], fy=[1][1], cx=[2][0], cy=[2][1].
    int u = (int)(intrinsics.columns[0][0] * x + intrinsics.columns[2][0]);
    // Camera-space +Y points upward, while image coordinates +Y point downward.
    int v = (int)(intrinsics.columns[2][1] - intrinsics.columns[1][1] * y);

    size_t width = CVPixelBufferGetWidthOfPlane(image, 0);
    size_t height = CVPixelBufferGetHeightOfPlane(image, 0);
    if (u < 0 || v < 0 || (size_t)u >= width || (size_t)v >= height) {
        return NO;
    }

    // Occlusion test: the LiDAR depth map (lower-res, same landscape sensor
    // orientation as the RGB image) gives the distance to the NEAREST real
    // surface at this pixel. If this vertex sits well behind that surface, the
    // camera could not actually see it here (it is hidden by nearer geometry),
    // so refuse to paint it — this stops a foreground object's pixels from
    // smearing onto walls/floor behind it. Compared in camera-space z, which is
    // <= the ray distance the depth map may report, so the bias is toward
    // keeping color rather than over-rejecting.
    if (depthMap != NULL) {
        size_t depthWidth = CVPixelBufferGetWidth(depthMap);
        size_t depthHeight = CVPixelBufferGetHeight(depthMap);
        int du = (int)((float)u * (float)depthWidth / (float)width);
        int dv = (int)((float)v * (float)depthHeight / (float)height);
        if (du >= 0 && dv >= 0 && (size_t)du < depthWidth && (size_t)dv < depthHeight) {
            const uint8_t *depthBase = (const uint8_t *)CVPixelBufferGetBaseAddress(depthMap);
            size_t depthRow = CVPixelBufferGetBytesPerRow(depthMap);
            float measured = *(const float *)(depthBase + (size_t)dv * depthRow + (size_t)du * sizeof(float));

            // Trust the depth only at medium/high confidence; otherwise skip the
            // occlusion test rather than risk wrongly rejecting a good sample.
            uint8_t conf = 2;  // ARConfidenceLevelHigh
            if (confidenceMap != NULL) {
                const uint8_t *confBase = (const uint8_t *)CVPixelBufferGetBaseAddress(confidenceMap);
                size_t confRow = CVPixelBufferGetBytesPerRow(confidenceMap);
                conf = confBase[(size_t)dv * confRow + (size_t)du];
            }

            float vertexDepth = -cameraPoint.z;
            if (conf >= 1 && measured > 0.0f && vertexDepth - measured > 0.12f) {
                return NO;  // occluded by nearer geometry
            }
        }
    }

    // Bi-planar YCbCr: plane 0 full-res luma, plane 1 half-res interleaved
    // Cb/Cr. Full-range BT.601 conversion (ARKit captures full-range YCbCr).
    const uint8_t *yBase = (const uint8_t *)CVPixelBufferGetBaseAddressOfPlane(image, 0);
    const uint8_t *cbcrBase = (const uint8_t *)CVPixelBufferGetBaseAddressOfPlane(image, 1);
    if (yBase == NULL || cbcrBase == NULL) {
        return NO;
    }
    size_t yRow = CVPixelBufferGetBytesPerRowOfPlane(image, 0);
    size_t cbcrRow = CVPixelBufferGetBytesPerRowOfPlane(image, 1);

    float Y = (float)yBase[(size_t)v * yRow + (size_t)u];
    const uint8_t *chroma = cbcrBase + ((size_t)(v / 2)) * cbcrRow + ((size_t)(u / 2)) * 2;
    float Cb = (float)chroma[0] - 128.0f;
    float Cr = (float)chroma[1] - 128.0f;

    outRGB[0] = [self clampByte:Y + 1.402f * Cr];
    outRGB[1] = [self clampByte:Y - 0.344136f * Cb - 0.714136f * Cr];
    outRGB[2] = [self clampByte:Y + 1.772f * Cb];
    return YES;
}

- (uint8_t)clampByte:(float)value {
    if (value < 0.0f) return 0;
    if (value > 255.0f) return 255;
    return (uint8_t)value;
}

@end
