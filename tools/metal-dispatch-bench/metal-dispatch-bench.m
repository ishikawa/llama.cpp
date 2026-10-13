#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <errno.h>
#include <limits.h>
#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    NSUInteger dispatches;
    NSUInteger barriers;
    NSUInteger elements;
    NSUInteger bindings;
    NSUInteger buffers;
    BOOL concurrent;
    const char * pattern;
    NSUInteger groups;
    NSUInteger every;
} bench_case;

typedef struct {
    double gpu_us;
    double span_us;
    double encode_us;
    double wall_us;
} bench_result;

static double now_us(void) {
    static mach_timebase_info_data_t info;
    if (!info.denom) {
        mach_timebase_info(&info);
    }
    return (double) mach_absolute_time() * info.numer / info.denom / 1000.0;
}

static int compare_double(const void * a, const void * b) {
    const double x = *(const double *) a, y = *(const double *) b;
    return (x > y) - (x < y);
}

static double median(double * values, NSUInteger count) {
    qsort(values, count, sizeof(*values), compare_double);
    return count % 2 ? values[count / 2] : (values[count / 2 - 1] + values[count / 2]) / 2;
}

static NSUInteger parse_positive(const char * value) {
    char * end;
    errno = 0;
    unsigned long long result = strtoull(value, &end, 10);
    if (errno || value[0] == '-' || !value[0] || *end || result == 0 || result > 1000000) {
        fprintf(stderr, "Expected an integer in [1, 1000000]: %s\n", value);
        exit(2);
    }
    return (NSUInteger) result;
}

static bench_result measure(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                            NSArray<id<MTLBuffer>> * data, bench_case c, NSUInteger reps) {
    double * samples = calloc(4 * reps, sizeof(double));
    if (!samples) {
        fprintf(stderr, "Cannot allocate samples\n");
        exit(1);
    }
    // One warmup per case; it is excluded from the medians.
    for (NSUInteger rep = 0; rep <= reps; ++rep) {
        @autoreleasepool {
            const double wall_start = now_us();
            double encode_us = 0;
            NSMutableArray<id<MTLCommandBuffer>> * buffers = [NSMutableArray array];
            NSUInteger global = 0;
            for (NSUInteger cb = 0; cb < c.buffers; ++cb) {
                const double encode_start = now_us();
                id<MTLCommandBuffer> command = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoderWithDispatchType:c.concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
                if (!command || !encoder) {
                    fprintf(stderr, "Cannot create command buffer/encoder\n");
                    exit(1);
                }
                [encoder setComputePipelineState:pipeline];
                const NSUInteger threads = c.elements ? MIN((NSUInteger) 256, pipeline.maxTotalThreadsPerThreadgroup) : 1;
                const uint32_t args[2] = {(uint32_t) c.elements, (uint32_t) (threads * c.groups)};
                const float scale = 1.00001f;
                for (NSUInteger j = 0; j < 4; ++j) {
                    [encoder setBuffer:data[j] offset:0 atIndex:j];
                }
                [encoder setBytes:args length:sizeof(args) atIndex:4];
                [encoder setBytes:&scale length:sizeof(scale) atIndex:5];
                const BOOL periodic = !strcmp(c.pattern, "periodic");
                const NSUInteger count = c.dispatches / c.buffers + (cb < c.dispatches % c.buffers);
                for (NSUInteger i = 0; i < count; ++i, ++global) {
                    if (c.bindings == 6) {
                        for (NSUInteger j = 0; j < 3; ++j) {
                            [encoder setBuffer:data[j] offset:0 atIndex:j];
                        }
                        [encoder setBytes:args length:sizeof(args) atIndex:4];
                        [encoder setBytes:&scale length:sizeof(scale) atIndex:5];
                    }
                    if (c.elements || c.bindings == 6) {
                        [encoder setBuffer:data[3] offset:global * c.elements * sizeof(float) atIndex:3];
                    }
                    [encoder dispatchThreadgroups:MTLSizeMake(c.groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
                    // Spread exactly B barriers over N dispatches, including encoder boundaries.
                    const BOOL barrier = periodic ?
                        (global + 1) % c.every == 0 :
                        ((global + 1) * c.barriers / c.dispatches != global * c.barriers / c.dispatches);
                    if (barrier) {
                        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    }
                }
                [encoder endEncoding];
                encode_us += now_us() - encode_start;
                [buffers addObject:command];
                [command commit];
            }
            double gpu_us = 0, first = INFINITY, last = 0;
            for (id<MTLCommandBuffer> command in buffers) {
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted || command.GPUStartTime <= 0 || command.GPUEndTime < command.GPUStartTime) {
                    fprintf(stderr, "Command buffer failed or GPU timestamps unavailable: %s\n", command.error.description.UTF8String);
                    exit(1);
                }
                gpu_us += 1e6 * (command.GPUEndTime - command.GPUStartTime);
                first = MIN(first, command.GPUStartTime);
                last = MAX(last, command.GPUEndTime);
            }
            if (rep) {
                samples[0 * reps + rep - 1] = gpu_us;
                samples[1 * reps + rep - 1] = 1e6 * (last - first);
                samples[2 * reps + rep - 1] = encode_us;
                samples[3 * reps + rep - 1] = now_us() - wall_start;
            }
        }
    }
    bench_result result = {median(samples, reps), median(samples + reps, reps), median(samples + 2 * reps, reps), median(samples + 3 * reps, reps)};
    free(samples);
    return result;
}

static void print_result(bench_case c, bench_result r, bench_result baseline, BOOL csv) {
    const double dispatch_us = r.gpu_us / c.dispatches;
    const double barrier_us = c.barriers ? (r.gpu_us - baseline.gpu_us) / c.barriers : 0;
    if (csv) {
        printf("%s,%s,%lu,%lu,%lu,%lu,%lu,%lu,%.3f,%.3f,%.3f,%.3f,%.6f,%.6f\n", c.pattern, c.concurrent ? "concurrent" : "serial",
               (unsigned long) c.elements, (unsigned long) c.dispatches, (unsigned long) c.barriers, (unsigned long) c.buffers, (unsigned long) c.bindings, (unsigned long) c.groups,
               r.gpu_us, r.span_us, r.encode_us, r.wall_us, dispatch_us, barrier_us);
    } else {
        printf("%-12s %-10s %6lu %6lu %6lu %2lu %2lu %2lu %10.3f %10.3f %10.3f %10.3f %10.6f %10.6f\n", c.pattern, c.concurrent ? "concurrent" : "serial",
               (unsigned long) c.elements, (unsigned long) c.dispatches, (unsigned long) c.barriers, (unsigned long) c.buffers, (unsigned long) c.bindings, (unsigned long) c.groups,
               r.gpu_us, r.span_us, r.encode_us, r.wall_us, dispatch_us, barrier_us);
    }
}

int main(int argc, const char ** argv) {
    @autoreleasepool {
        NSUInteger n = 3428, reps = 11, every = 4;
        BOOL csv = NO;
        for (int i = 1; i < argc; ++i) {
            if (!strcmp(argv[i], "--csv")) {
                csv = YES;
            } else if (!strcmp(argv[i], "--help")) {
                printf("Usage: %s [--n N] [--reps R] [--every K] [--csv]\n", argv[0]);
                puts("Default: N=3428, R=11, K=4. Model cases always use N=3428, B=1992, two command buffers/encoders.\n"
                     "Each case has one excluded warmup; --n is limited to 13107 by the disjoint output allocation.");
                return 0;
            } else if ((!strcmp(argv[i], "--n") || !strcmp(argv[i], "--reps") || !strcmp(argv[i], "--every")) && i + 1 < argc) {
                const char * option = argv[i++];
                NSUInteger value = parse_positive(argv[i]);
                if (!strcmp(option, "--n")) n = value;
                else if (!strcmp(option, "--reps")) reps = value;
                else every = value;
            } else {
                fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
                return 2;
            }
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSString * source = @"#include <metal_stdlib>\n"
                            "using namespace metal;\n"
                            "kernel void empty_kernel(uint i [[thread_position_in_grid]]) { (void)i; }\n"
                            "struct Args { uint n; uint stride; };\n"
                            "kernel void elementwise(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], "
                            "device const float *c [[buffer(2)]], device float *out [[buffer(3)]], "
                            "constant Args &args [[buffer(4)]], constant float &s [[buffer(5)]], uint i [[thread_position_in_grid]]) { "
                            "for (; i < args.n; i += args.stride) out[i] = (a[i] + b[i]) * s + c[i]; }\n";
        NSError * error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library || !queue) {
            fprintf(stderr, "Metal initialization failed: %s\n", error.description.UTF8String);
            return 1;
        }
        const NSUInteger slots = MAX(n, (NSUInteger) 3428);
        if (slots > (512 * 1024 * 1024) / (10240 * sizeof(float))) {
            fprintf(stderr, "--n exceeds the 512 MiB disjoint output limit (13107 dispatches)\n");
            return 2;
        }
        NSMutableArray<id<MTLBuffer>> * data = [NSMutableArray array];
        for (NSUInteger j = 0; j < 4; ++j) {
            id<MTLBuffer> buffer = [device newBufferWithLength:(j == 3 ? slots : 1) * 10240 * sizeof(float) options:MTLResourceStorageModeShared];
            if (!buffer) return 1;
            float * values = buffer.contents;
            if (j < 3) {
                for (NSUInteger i = 0; i < 10240; ++i) values[i] = 0.25f + (float) j;
            }
            [data addObject:buffer];
        }
        if (csv) {
            puts("pattern,encoder,elements,dispatches,barriers,command_buffers,bindings_per_dispatch,threadgroups_per_dispatch,gpu_sum_us,gpu_span_us,encode_us,wall_us,gpu_us_per_dispatch,delta_gpu_us_per_barrier");
        } else {
            printf("Device: %s; medians of %lu repeats after one warmup per case. All times in us.\n", device.name.UTF8String, (unsigned long) reps);
            puts("pattern      encoder      elems      N      B CB IO TG     GPUsum    GPUspan     encode       wall     GPU/N    delta/B");
            puts("IO=6 means four setBuffer and two setBytes per dispatch; IO=1 only changes the output slice; delta/B uses the matching B=0 case.\n"
                 "Elementwise dispatches have disjoint output slices and immutable inputs; allocation and pipeline compilation are excluded.\n"
                 "GPU/N includes kernel work; delta/B includes serialization changes and can be negative.\n"
                 "Metal ignores memoryBarrierWithScope on serial encoders; their delta/B is a control measurement.");
        }
        const NSUInteger sizes[] = {0, 2560, 10240, 2560};
        for (NSUInteger kind = 0; kind < 4; ++kind) {
            id<MTLFunction> function = [library newFunctionWithName:kind ? @"elementwise" : @"empty_kernel"];
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) {
                fprintf(stderr, "Pipeline failed: %s\n", error.description.UTF8String);
                return 1;
            }
            for (NSUInteger concurrent = 0; concurrent < 2; ++concurrent) {
                for (NSUInteger bindings = 0; bindings < 2; ++bindings) {
                    const NSUInteger threads = kind ? MIN((NSUInteger) 256, pipeline.maxTotalThreadsPerThreadgroup) : 1;
                    const NSUInteger groups = kind == 3 ? 1 : (kind ? (sizes[kind] + threads - 1) / threads : 1);
                    bench_case c = {n, 0, sizes[kind], bindings ? 6 : (kind ? 1 : 0), 1, concurrent, "none", groups, every};
                    bench_result baseline = measure(queue, pipeline, data, c, reps);
                    print_result(c, baseline, baseline, csv);
                    c.barriers = n; c.pattern = "every";
                    print_result(c, measure(queue, pipeline, data, c, reps), baseline, csv);
                    c.barriers = n / every; c.pattern = "periodic";
                    print_result(c, measure(queue, pipeline, data, c, reps), baseline, csv);
                    // Keep the measured model topology even when --n changes the sweep above.
                    c.dispatches = 3428; c.barriers = 0; c.buffers = 2; c.pattern = "model-base";
                    baseline = measure(queue, pipeline, data, c, reps);
                    print_result(c, baseline, baseline, csv);
                    c.barriers = 1992; c.pattern = "model";
                    print_result(c, measure(queue, pipeline, data, c, reps), baseline, csv);
                }
            }
        }
        return 0;
    }
}
