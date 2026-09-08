#import "ScannerViewController.h"
#import "LiveMeshStreamer.h"
#import <ARKit/ARKit.h>
#import <SceneKit/SceneKit.h>

// Default host/port. Edit the field at runtime; the port matches the host's
// Connection defaults; the operator can change the host in the app.
static NSString *const kDefaultHost = @"192.168.0.238";
static const uint16_t kDefaultPort = 9099;

@interface ScannerViewController () <ARSessionDelegate>
@property (nonatomic, strong) ARSCNView *sceneView;
@property (nonatomic, strong) LiveMeshStreamer *streamer;
@property (nonatomic, strong) UITextField *hostField;
@property (nonatomic, strong) UIButton *connectButton;
@property (nonatomic, strong) UIButton *scanButton;
@property (nonatomic, strong) UIButton *resetButton;
@property (nonatomic, strong) UILabel *statusLabel;
@end

@implementation ScannerViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor blackColor];

    self.sceneView = [[ARSCNView alloc] initWithFrame:self.view.bounds];
    self.sceneView.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    self.sceneView.session.delegate = self;
    // Show ARKit's world origin over the camera preview.
    self.sceneView.debugOptions = ARSCNDebugOptionShowWorldOrigin;
    [self.view addSubview:self.sceneView];

    self.streamer = [[LiveMeshStreamer alloc] init];
    // cuVSLAM pipeline: stream continuous posed RGB-D so the host runs cuVSLAM
    // RGB-D odometry (drift-corrected) instead of trusting ARKit's pose. Set to
    // NO to fall back to motion-gated keyframes for the offline texture bake.
    self.streamer.continuousRGBD = YES;
    // Host -> phone remote control: the host can start/stop the scan over the
    // socket (MSG_CONTROL_START_SCAN = 8, STOP = 9). The block is delivered on
    // the main queue, so it can touch UI / the AR session directly.
    __weak typeof(self) weakSelf = self;
    self.streamer.onRemoteControl = ^(uint8_t messageType) {
        typeof(self) strongSelf = weakSelf;
        if (strongSelf == nil) {
            return;
        }
        if (messageType == 8) {
            [strongSelf startScanning];
        } else if (messageType == 9) {
            [strongSelf stopScanning];
        }
    };
    // Launch idle: wait for an explicit Start (button or host remote control)
    // rather than streaming the moment we connect. Matches the "re-arm with
    // Start Scan" model already used after backgrounding.
    self.streamer.streamingEnabled = NO;

    [self buildControls];

    // Auto-stop streaming when the app is backgrounded or closed, so the final
    // burst of frames as the phone is set down / the app exits can't repaint the
    // finished scan. The user re-arms with the Start Scan button.
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(appWillResignActive)
                                                 name:UIApplicationWillResignActiveNotification
                                               object:nil];
}

- (void)dealloc {
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (void)buildControls {
    CGFloat top = 60.0;
    CGFloat margin = 16.0;
    CGFloat width = self.view.bounds.size.width - 2 * margin;

    self.hostField = [[UITextField alloc] initWithFrame:CGRectMake(margin, top, width - 110, 40)];
    self.hostField.autoresizingMask = UIViewAutoresizingFlexibleWidth;
    self.hostField.borderStyle = UITextBorderStyleRoundedRect;
    self.hostField.backgroundColor = [UIColor colorWithWhite:1.0 alpha:0.85];
    self.hostField.text = kDefaultHost;
    self.hostField.placeholder = @"host ip";
    self.hostField.keyboardType = UIKeyboardTypeNumbersAndPunctuation;
    self.hostField.autocorrectionType = UITextAutocorrectionTypeNo;
    self.hostField.autocapitalizationType = UITextAutocapitalizationTypeNone;
    [self.view addSubview:self.hostField];

    self.connectButton = [UIButton buttonWithType:UIButtonTypeSystem];
    self.connectButton.frame = CGRectMake(self.view.bounds.size.width - margin - 94, top, 94, 40);
    self.connectButton.autoresizingMask = UIViewAutoresizingFlexibleLeftMargin;
    self.connectButton.backgroundColor = [UIColor colorWithWhite:1.0 alpha:0.85];
    self.connectButton.layer.cornerRadius = 6.0;
    [self.connectButton setTitle:@"Connect" forState:UIControlStateNormal];
    [self.connectButton addTarget:self
                           action:@selector(connectTapped)
                 forControlEvents:UIControlEventTouchUpInside];
    [self.view addSubview:self.connectButton];

    self.statusLabel = [[UILabel alloc] initWithFrame:CGRectMake(margin, top + 48, width, 60)];
    self.statusLabel.autoresizingMask = UIViewAutoresizingFlexibleWidth;
    self.statusLabel.numberOfLines = 0;
    self.statusLabel.textColor = [UIColor whiteColor];
    self.statusLabel.font = [UIFont monospacedDigitSystemFontOfSize:13 weight:UIFontWeightRegular];
    self.statusLabel.text = @"disconnected";
    [self.view addSubview:self.statusLabel];

    self.scanButton = [UIButton buttonWithType:UIButtonTypeSystem];
    self.scanButton.frame = CGRectMake(margin, top + 112, width, 44);
    self.scanButton.autoresizingMask = UIViewAutoresizingFlexibleWidth;
    self.scanButton.backgroundColor = [UIColor colorWithWhite:1.0 alpha:0.85];
    self.scanButton.layer.cornerRadius = 6.0;
    self.scanButton.titleLabel.font = [UIFont boldSystemFontOfSize:17];
    [self.scanButton addTarget:self
                        action:@selector(scanTapped)
              forControlEvents:UIControlEventTouchUpInside];
    [self.view addSubview:self.scanButton];
    [self updateScanButton];

    // Reset Position: zero the twin origin to the phone's current (mounted) pose.
    self.resetButton = [UIButton buttonWithType:UIButtonTypeSystem];
    self.resetButton.frame = CGRectMake(margin, top + 164, width, 44);
    self.resetButton.autoresizingMask = UIViewAutoresizingFlexibleWidth;
    self.resetButton.backgroundColor = [UIColor colorWithWhite:1.0 alpha:0.85];
    self.resetButton.layer.cornerRadius = 6.0;
    self.resetButton.titleLabel.font = [UIFont boldSystemFontOfSize:17];
    [self.resetButton setTitle:@"Reset Position" forState:UIControlStateNormal];
    [self.resetButton addTarget:self
                         action:@selector(resetPositionTapped)
               forControlEvents:UIControlEventTouchUpInside];
    [self.view addSubview:self.resetButton];
}

// Reflect the streamer's gate on the button: it shows the action it will take.
- (void)updateScanButton {
    BOOL scanning = self.streamer.streamingEnabled;
    [self.scanButton setTitle:(scanning ? @"Stop Scan" : @"Start Scan")
                     forState:UIControlStateNormal];
    [self.scanButton setTitleColor:(scanning ? [UIColor systemRedColor] : [UIColor systemGreenColor])
                          forState:UIControlStateNormal];
}

- (void)scanTapped {
    if (self.streamer.streamingEnabled) {
        [self stopScanning];
    } else {
        [self startScanning];
    }
}

// Re-anchor the twin origin to the phone's current pose. Use after mounting the
// phone on the rover so the rover's twin start pose matches where it actually is.
- (void)resetPositionTapped {
    if (!self.streamer.connected) {
        self.statusLabel.text = @"Reset Position: connect to the host first";
        return;
    }
    [self.streamer sendRealign];
    self.statusLabel.text = @"Reset Position: re-anchored twin origin to current pose";
}

// Shared start/stop, used by both the on-screen button and host remote control
// (onRemoteControl). Must run on the main thread.
- (void)startScanning {
    self.streamer.streamingEnabled = YES;
    [self updateScanButton];
}

- (void)stopScanning {
    // Stopping the scan is the explicit "I'm done" signal: ask the host to bake
    // the accumulated mesh + keyframes into a textured mesh.
    if (self.streamer.streamingEnabled) {
        self.streamer.streamingEnabled = NO;
        [self.streamer sendFinalize];
    }
    [self updateScanButton];
}

- (void)appWillResignActive {
    self.streamer.streamingEnabled = NO;
    [self updateScanButton];
}

- (void)viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];

    if (![ARWorldTrackingConfiguration supportsSceneReconstruction:ARSceneReconstructionMesh]) {
        self.statusLabel.text = @"This device has no LiDAR scene reconstruction.";
        return;
    }
    ARWorldTrackingConfiguration *config = [[ARWorldTrackingConfiguration alloc] init];
    // In continuous RGB-D (nvblox) mode the host builds geometry from the depth
    // stream and ignores ARKit mesh anchors, so skip scene reconstruction (and
    // environment texturing) entirely. Otherwise the phone spends GPU building a
    // mesh and floods the shared serial send queue with MSG_UPSERT_ANCHOR frames
    // that were measured at ~36% of link bandwidth -- all discarded by the host.
    // LiDAR depth comes from frameSemantics below, independent of reconstruction.
    if (!self.streamer.continuousRGBD) {
        config.sceneReconstruction = ARSceneReconstructionMesh;
        config.environmentTexturing = AREnvironmentTexturingAutomatic;
    }
    // Per-frame LiDAR depth: the RGB-D stream's depth map, and (old path) used to
    // occlusion-test vertex coloring so foreground objects don't paint onto
    // geometry behind them. Prefer the temporally smoothed depth when available.
    if ([ARWorldTrackingConfiguration supportsFrameSemantics:ARFrameSemanticSmoothedSceneDepth]) {
        config.frameSemantics |= ARFrameSemanticSmoothedSceneDepth;
    } else if ([ARWorldTrackingConfiguration supportsFrameSemantics:ARFrameSemanticSceneDepth]) {
        config.frameSemantics |= ARFrameSemanticSceneDepth;
    }
    [self.sceneView.session runWithConfiguration:config
                                         options:ARSessionRunOptionResetTracking |
                                                 ARSessionRunOptionRemoveExistingAnchors];

    // Auto-connect on launch so the host can drive start/stop remotely without a
    // manual Connect tap. The user can still edit the host field + tap Connect to
    // point elsewhere (which reconnects).
    if (!self.streamer.connected) {
        NSString *host = self.hostField.text.length ? self.hostField.text : kDefaultHost;
        [self.streamer connectToHost:host port:kDefaultPort];
    }

    [self scheduleStatusUpdates];
}

- (void)connectTapped {
    [self.hostField resignFirstResponder];
    NSString *host = self.hostField.text.length ? self.hostField.text : kDefaultHost;
    [self.streamer connectToHost:host port:kDefaultPort];
}

- (void)scheduleStatusUpdates {
    [NSTimer scheduledTimerWithTimeInterval:0.5
                                     repeats:YES
                                       block:^(NSTimer *timer) {
        self.statusLabel.text = [NSString stringWithFormat:@"%@  %@  frames: %lu  keyframes: %lu",
                                 self.streamer.connected ? @"connected" : @"disconnected",
                                 self.streamer.streamingEnabled ? @"scanning" : @"stopped",
                                 (unsigned long)self.streamer.framesSent,
                                 (unsigned long)self.streamer.keyframesSent];
    }];
}

#pragma mark - ARSessionDelegate

// Anchors arrive on a private ARKit queue, not the main thread.
- (void)session:(ARSession *)session didAddAnchors:(NSArray<ARAnchor *> *)anchors {
    [self streamMeshAnchors:anchors frame:session.currentFrame];
}

- (void)session:(ARSession *)session didUpdateAnchors:(NSArray<ARAnchor *> *)anchors {
    [self streamMeshAnchors:anchors frame:session.currentFrame];
}

// Per-frame hook: capture posed keyframes for the offline bake as the camera
// moves around the room. The streamer throttles + motion-gates internally.
- (void)session:(ARSession *)session didUpdateFrame:(ARFrame *)frame {
    [self.streamer sendPhonePoseForFrame:frame];
    [self.streamer considerKeyframeForFrame:frame];
}

- (void)session:(ARSession *)session didRemoveAnchors:(NSArray<ARAnchor *> *)anchors {
    // Preserve removed ARMeshAnchors on the host so the Isaac room scan builds
    // up as you walk around. ARKit may retire anchors behind the camera as it
    // refines the reconstruction; forwarding those removals makes the visible
    // twin look like it is erasing itself. A future "dynamic object" mode can
    // re-enable remove_anchor messages explicitly.
    (void)session;
    (void)anchors;
}

- (void)streamMeshAnchors:(NSArray<ARAnchor *> *)anchors frame:(ARFrame *)frame {
    if (frame == nil) {
        return;
    }
    for (ARAnchor *anchor in anchors) {
        if ([anchor isKindOfClass:[ARMeshAnchor class]]) {
            [self.streamer handleMeshAnchor:(ARMeshAnchor *)anchor frame:frame];
        }
    }
}

@end
