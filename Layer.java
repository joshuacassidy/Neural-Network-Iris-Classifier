import java.util.Arrays;
import java.util.Random;

public class Layer {

    private float[] output, input, weights, dWeights;
    private Random random;

    public Layer(int inputSize, int outputSize) {
        this.output = new float[outputSize];
        this.input = new float[inputSize+1];
        this.weights = new float[(inputSize+1) * outputSize];
        this.dWeights = new float[weights.length];
        this.random = new Random();
        initWeights();
    }

    private void initWeights() {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (random.nextFloat()-0.5f) * 4f;
        }
    }


    public float[] run(float[] inputActivations) {
        System.arraycopy(inputActivations, 0, input, 0, inputActivations.length);
        // bias node
        input[input.length-1] = 1;
        int offset = 0;

        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < input.length; j++) {
                output[i] += weights[offset+j] * input[j];
            }
            output[i] = ActivationFunction.sigmoid(output[i]);
            offset += input.length;
        }
        return Arrays.copyOf(output, output.length);
    }

    public float[] train(float[] error, float learningRate, float momentum) {
        // tracking the edge weight values for the given neurons
        int offset =0;
        float[] nextError = new float[input.length];

        for (int i = 0; i < output.length; i++) {
            // already calculated the sigmoid value at line 38
            float delta = error[i] * ActivationFunction.dSigmoid(output[i]);

            for (int j = 0; j < input.length; j++) {
                int weightIndex = offset+j;
                nextError[j] = nextError[j] + weights[weightIndex] * delta;
                float dw = input[j] * delta * learningRate;
                weights[weightIndex] += dWeights[weightIndex] * momentum + dw;
                dWeights[weightIndex] = dw;
            }

            offset += input.length;

        }

        return nextError;

    }
}
