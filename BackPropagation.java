import java.util.HashMap;
import java.util.Map;

public class BackPropagation {

    private Layer[] layers;

    public BackPropagation(int inputSize, int hiddenSize, int outputSize) {
        this.layers = new Layer[2];
        layers[0] = new Layer(inputSize, hiddenSize);
        layers[1] = new Layer(hiddenSize, outputSize);
    }

    public Layer[] getLayers() {
        return layers;
    }

    public float[] run(float[] input) {

        float[] inputActivations = input;
        for (int i = 0; i < layers.length; i++) {
            inputActivations = layers[i].run(inputActivations);
        }

        return inputActivations;
    }

    public String test(float[] input) {

        float[] inputActivations = input;
        for (int i = 0; i < layers.length; i++) {
            inputActivations = layers[i].run(inputActivations);
        }

        for (int i = 0; i < inputActivations.length; i++) {
            inputActivations[i] = Math.round(inputActivations[i]);
        }
        
        HashMap<Float[], String> hashMap = new HashMap<>();
        hashMap.put(new Float[] {1f,0f,0f}, "Iris setosa");
        hashMap.put(new Float[] {0f,1f,0f}, "Iris versicolor");
        hashMap.put(new Float[] {0f,0f,1f}, "Iris virginica");

        for(Map.Entry<Float[], String> entry : hashMap.entrySet()){
            Float[] key=entry.getKey();
            String val=entry.getValue();
            int i;
            for (i = 0; i < key.length; i++) {
                if (inputActivations[i] != key[i]) {
                    break;
                }
            }
            if (i==key.length) {
                return val;
            }
        }


        return null;
    }

    public void train(float[] input, float[] targetOutput, float learningRate, float momentum) {
        float[] calculatedOutput = run(input);
        float[] error = new float[calculatedOutput.length];

        for (int i = 0; i < error.length; i++) {
            error[i] = targetOutput[i] - calculatedOutput[i];
        }

        for (int i = layers.length-1; i >= 0; i--) {
            error = layers[i].train(error, learningRate, momentum);
        }

    }


}
