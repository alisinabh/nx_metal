#import <Metal/Metal.h>

@interface MTLDeviceReference : NSObject

@property (strong, nonatomic) id<MTLDevice> device;

- (instancetype)initWithDevice:(id<MTLDevice>)device;

@end
