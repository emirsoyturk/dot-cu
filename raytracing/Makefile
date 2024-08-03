NVCC = /usr/local/cuda/bin/nvcc
CFLAGS = -Iinclude `pkg-config --cflags opencv4` -diag-suppress=611
LDFLAGS = `pkg-config --libs opencv4`
SRCS = src/main.cu src/raytracer.cu
TARGET = raytracing

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)