import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) {

        BackPropagation backPropagation = new BackPropagation(4, 5, 3);

        float[][] trainingData;

        float[][] trainingResults;

        Scanner s;

        int rowCount = 0;

        HashMap<Integer, String> oneRow = new HashMap<Integer, String>();

        try {
            s = new Scanner(new File("src/iris.data.txt"));
            while (s.hasNext()) {

                oneRow.put(rowCount, s.next());

                rowCount++;

            }

            s.close();

            trainingData = new float[rowCount][4];

            trainingResults = new float[rowCount][3];

            for (int i = 0; i < rowCount; i++) {

                String[] oneRowSplit = oneRow.get(i).split(",");

                for (int j = 0; j < 4; j++) {

                    trainingData[i][j] = Float.valueOf(oneRowSplit[j]);

                }

                if (oneRowSplit[4].equals("Iris-setosa")) {

                    trainingResults[i] = new float[] { 1f, 0f, 0f };

                }

                if (oneRowSplit[4].equals("Iris-versicolor")) {

                    trainingResults[i] = new float[] { 0f, 1f, 0f };

                }

                if (oneRowSplit[4].equals("Iris-virginica")) {

                    trainingResults[i] = new float[] { 0f, 0f, 1f };

                }

            }

            for (int iterations = 0; iterations < Constants.ITERATIONS; iterations++) {

                for (int i = 0; i < trainingResults.length; i++) {
                    backPropagation.train(trainingData[i], trainingResults[i], Constants.LEARNING_RATE, Constants.MOMENTUM);
                }

                if (iterations%5000==0) {
                    System.out.println("Training...");
                    System.out.println();


                }
            }


            System.out.println(backPropagation.test(new float[] {4.9f,3.1f,1.5f,0.1f})); // Should evaluate to: Iris setosa
            System.out.println(backPropagation.test(new float[] {6.3f,2.7f,4.9f,1.8f})); // Should evaluate to: Iris-virginica
            System.out.println(backPropagation.test(new float[] {5.7f,2.9f,4.2f,1.3f})); // Should evaluate to: Iris-versicolor
            System.out.println(backPropagation.test(new float[] {5.8f,2.7f,4.1f,1.0f})); // Should evaluate to: Iris-versicolor

        } catch (FileNotFoundException e) {
            System.out.println("Error!!! File could not be found");
        }

    }

}
