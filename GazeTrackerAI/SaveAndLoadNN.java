import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class SaveAndLoadNN {

    GazeTracker gTracker;

    public SaveAndLoadNN(GazeTracker gazeTracker) {
        this.gTracker = gazeTracker;
    }
    
    // Сохранение модели в файл
    public void saveWeights(String filename, GazeTracker network) {
        try (PrintWriter out = new PrintWriter(new FileWriter(filename))) {
            // Сохранение весов от входного к скрытому слою X
            System.out.println("Save to file weights.txt input-hidden weights");
            for (float[] layer : gTracker.inputToHiddenWeightsX) {
                for (float weight : layer) {
                    out.print(weight + " ");
                }
                out.println();
            }

            // Сохранение весов от входного к скрытому слою Y
            System.out.println("Save to file weights.txt input-hidden weights");
            for (float[] layer : gTracker.inputToHiddenWeightsY) {
                for (float weight : layer) {
                    out.print(weight + " ");
                }
                out.println();
            }

            // Сохранение весов для X от скрытого к выходному слою 
            System.out.println("Save to file weights.txt hidden-output X weights");
            for (float[] layer : gTracker.hiddenToOutputWeightsX) {
                for (float weight : layer) {
                    out.print(weight + " ");
                }
                out.println();
            }

            // Сохранение весов для Y от скрытого к выходному слою 
            System.out.println("Save to file weights.txt hidden-output Y weights");
            for (float[] layer : gTracker.hiddenToOutputWeightsY) {
                for (float weight : layer) {
                    out.print(weight + " ");
                }
                out.println();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    // Загрузка модели из файла
    public void loadWeights(String filename) {
        File file = new File(filename);
        if (file.length() == 0) {
            // Инициализация весов с помощью Math.random(), если файл пустой
            gTracker.initializeWeights();
        } else {
            try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
                String line;
                int i = 0;
                // Загрузка весов от входного к скрытому слою
                System.out.println("Read from file weights.txt input-hidden X weights");
                while ((line = br.readLine()) != null && i < gTracker.inputToHiddenWeightsX.length) {
                    String[] weights = line.trim().split("\\s+");
                    for (int j = 0; j < weights.length; j++) {
                        gTracker.inputToHiddenWeightsX[i][j] = Float.parseFloat(weights[j]);
                    }
                    i++;
                }
                i = 0;
                // Загрузка весов от входного к скрытому слою
                System.out.println("Read from file weights.txt input-hidden Y weights");
                while ((line = br.readLine()) != null && i < gTracker.inputToHiddenWeightsY.length) {
                    String[] weights = line.trim().split("\\s+");
                    for (int j = 0; j < weights.length; j++) {
                        gTracker.inputToHiddenWeightsY[i][j] = Float.parseFloat(weights[j]);
                    }
                    i++;
                }
                i = 0;
                // Загрузка весов для X от скрытого к выходному слою
                System.out.println("Read from file weights.txt hidden-output X weights");
                while ((line = br.readLine()) != null && i < gTracker.hiddenToOutputWeightsX.length) {
                    String[] weights = line.trim().split("\\s+");
                    for (int j = 0; j < weights.length; j++) {
                        gTracker.hiddenToOutputWeightsX[i][j] = Float.parseFloat(weights[j]);
                    }
                    i++;
                }
                // Загрузка весов для Y от скрытого к выходному слою
                System.out.println("Read from file weights.txt hidden-output Y weights");
                while ((line = br.readLine()) != null && i < gTracker.hiddenToOutputWeightsY.length) {
                    String[] weights = line.trim().split("\\s+");
                    for (int j = 0; j < weights.length; j++) {
                        gTracker.hiddenToOutputWeightsY[i][j] = Float.parseFloat(weights[j]);
                    }
                    i++;
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
