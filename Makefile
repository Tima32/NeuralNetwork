NEURAL_NETWORK_TARGET=neural-network.elf

NEURAL_NETWORK_SOURCES=NeuralNetwork/main.cpp NeuralNetwork/Net.cpp

CC=$(CROSS_COMPILE)g++

NEURAL_NETWORK_OBJECTS=$(patsubst %.cpp,%.o,$(NEURAL_NETWORK_SOURCES))
all: $(NEURAL_NETWORK_TARGET)

%.o: %.cpp
	$(CC) -c $^ -o $@

$(NEURAL_NETWORK_TARGET): $(NEURAL_NETWORK_OBJECTS)
	$(CC) $^ -o $@ -pthread -lrt



clean:
	rm -f $(NEURAL_NETWORK_TARGET) $(NEURAL_NETWORK_OBJECTS)

.PHONY: all clean
