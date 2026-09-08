#import "AppDelegate.h"
#import "ScannerViewController.h"

@implementation AppDelegate

- (BOOL)application:(UIApplication *)application
    didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    // Fully programmatic window/root view controller: no storyboard or scene
    // manifest, because asset/IB compilation is unavailable under Theos/Linux.
    self.window = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
    self.window.rootViewController = [[ScannerViewController alloc] init];
    [self.window makeKeyAndVisible];
    // Scene reconstruction + streaming is hot; keep the screen awake.
    application.idleTimerDisabled = YES;
    return YES;
}

@end
