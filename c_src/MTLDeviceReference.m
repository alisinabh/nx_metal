#import "MTLDeviceReference.h"

@implementation MTLDeviceReference

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        _device = device;
    }
    return self;
}

@end

