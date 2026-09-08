#import <Foundation/Foundation.h>
#import <ARKit/ARKit.h>

NS_ASSUME_NONNULL_BEGIN

// Encodes length-prefixed RGB-D frames for the nvblox host mapper.
// Message layouts are defined in src/digital_twin/live_scan.py.
@interface LiveMeshStreamer : NSObject

// Minimum seconds between sends for the same anchor id (bandwidth throttle).
@property (nonatomic) NSTimeInterval minimumAnchorInterval;
@property (nonatomic, readonly) BOOL connected;
@property (nonatomic, readonly) NSUInteger framesSent;
@property (nonatomic, readonly) NSUInteger keyframesSent;

// Master gate for capturing/streaming. When NO, handleMeshAnchor: does nothing
// (the socket stays open and prior geometry/color is preserved on the host).
// Toggle this from a Stop/Start control and turn it off when the app resigns
// active, so lowering the phone or closing the app can't repaint the finished
// scan with near/dark close-up frames. Defaults to YES.
@property (nonatomic) BOOL streamingEnabled;

// Diagnostic visual-inertial capture mode. Defaults to NO. The host enables this
// only for raw scanlog recordings intended for cuVSLAM tests; normal nvblox scans
// keep CoreMotion off to avoid adding a 100 Hz stream.
@property (nonatomic) BOOL imuStreamingEnabled;

// Continuous RGB-D mode: stream posed RGB-D keyframes (~10 fps, no motion gate)
// for the host nvblox mapper. Defaults to NO (motion-gated keyframes for the
// offline bake).
@property (nonatomic) BOOL continuousRGBD;

// Host -> phone remote control. The host can push start/stop-scan commands over
// the same socket; this block
// is invoked on the main queue with messageType 8 (START) or 9 (STOP) so the UI
// can begin/finish a scan without a button tap. STOP should send finalize.
@property (nonatomic, copy, nullable) void (^onRemoteControl)(uint8_t messageType);

- (void)connectToHost:(NSString *)host port:(uint16_t)port;
- (void)disconnect;

// Throttles, encodes, and sends one mesh anchor. Safe to call from the ARKit
// session delegate queue.
- (void)handleMeshAnchor:(ARMeshAnchor *)anchor frame:(ARFrame *)frame;

// Captures a posed keyframe (JPEG + camera pose + intrinsics) for the offline
// bake when the camera has moved enough since the last one. Call once per AR
// frame. No-op while not streaming, disconnected, or the frame is unusable.
- (void)considerKeyframeForFrame:(ARFrame *)frame;

// Streams the continuous rover-localization pose (MSG_PHONE_POSE): T_A_C plus
// tracking state and a sequence number. Call once per AR frame. Sent whenever
// connected (independently of streamingEnabled) so localization continues even
// while mesh streaming is paused; internally throttled.
- (void)sendPhonePoseForFrame:(ARFrame *)frame;

// Signals the end of the scan so the mapper can save its final map revision.
- (void)sendFinalize;

// Tells the host to re-anchor the ARKit->twin alignment to the phone's current
// pose. Send this from Reset Position after mounting the phone on the rover.
- (void)sendRealign;

// Sends a remove_anchor message so the host drops the prim/grid cells.
- (void)removeAnchorWithId:(NSUUID *)anchorId;

@end

NS_ASSUME_NONNULL_END
