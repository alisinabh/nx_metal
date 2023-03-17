#import "MTLBufferReference.h"

@implementation MTLBufferReference

- (instancetype)initWithBuffer:(id<MTLBuffer>)buffer
{
    self = [super init];
    if (self)
    {
        _buffer = buffer;
    }
    return self;
}

@end
