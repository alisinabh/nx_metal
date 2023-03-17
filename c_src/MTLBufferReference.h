#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface MTLBufferReference : NSObject

@property (nonatomic, strong) id<MTLBuffer> buffer;

- (instancetype)initWithBuffer:(id<MTLBuffer>)buffer;

@end
