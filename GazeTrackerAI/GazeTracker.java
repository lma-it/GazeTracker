import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Random;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

import javax.imageio.ImageIO;


public class GazeTracker {

    // Параметры для размера слоев
    /**
     * Количество нейронов входного слоя = 240 * 320
     */
    public static int INPUT_NEURONS = 320 * 480;
    /**
     * Количество нейронов скрытого слоя X = 384. 
     */
    public static int HIDDEN_NEURONSX = 700;
    /**
     * Количество нейронов скрытого слоя Y = 384. 
     */
    public static int HIDDEN_NEURONSY = 700;
    /**
     * Количество нейронов выходного слоя X значения. 1 нейрон
     */
    public static int OUTPUT_NEURONSX = 1;
    /**
     * Количество нейронов выходного слоя Y значения. 1 нейрон
     */
    public static int OUTPUT_NEURONSY = 1;
    /**
     * Ширина экрана в пикселях = 1080.
     */
    public static int SCREEN_WIDTH  = 1080; //ScreenSizeGetter.getScreenWidth();
    /**
     * Высота экрана в пикселях = 2200.
     */
    public static int SCREEN_HEIGTH = 2200; //ScreenSizeGetter.getScreenHeight();
    
    // Веса для слоев
    /**
     * Веса между входным и скрытым слоем для X слоя. Количество = (240 * 320) * 384
     */
    public float[][] inputToHiddenWeightsX = new float[INPUT_NEURONS][HIDDEN_NEURONSX];
    /**
     * Веса между входным и скрытым слоем для Y слоя. Количество = (240 * 320) * 384
     */
    public float[][] inputToHiddenWeightsY = new float[INPUT_NEURONS][HIDDEN_NEURONSY];
    /**
     * Веса между скрытым слоем X и выходом X. Количество = 384 * 1
     */
    public float[][] hiddenToOutputWeightsX = new float[HIDDEN_NEURONSX][OUTPUT_NEURONSX];
    /**
     * Веса между скрытым слоем Y и выходом Y. Количество = 384 * 1
     */
    public float[][] hiddenToOutputWeightsY = new float[HIDDEN_NEURONSY][OUTPUT_NEURONSY];

    // Слои нейронов
    public float[] inputLayer = new float[INPUT_NEURONS];
    public float[] hiddenLayerX = new float[HIDDEN_NEURONSX];
    public float[] hiddenLayerY = new float[HIDDEN_NEURONSY];
    public float[] outputLayerX = new float[OUTPUT_NEURONSX];
    public float[] outputLayerY = new float[OUTPUT_NEURONSY];

    public float[] outputGradientsX = new float[OUTPUT_NEURONSX];
    public float[] outputGradientsY = new float[OUTPUT_NEURONSY];
    public float[] hiddenGradientsX = new float[HIDDEN_NEURONSX];
    public float[] hiddenGradientsY = new float[HIDDEN_NEURONSY];

    public float[] tempOutputX = new float[1];
    public float[] tempOutputY = new float[1];
    
    public float[][] tempHiddenToOutputX = new float[HIDDEN_NEURONSX][OUTPUT_NEURONSX];
    public float[][] tempHiddenToOutputY = new float[HIDDEN_NEURONSY][OUTPUT_NEURONSY];

    public float[][] tempInputToHiddenXWeights = new float[INPUT_NEURONS][HIDDEN_NEURONSX];
    public float[][] tempInputToHiddenYWeights = new float[INPUT_NEURONS][HIDDEN_NEURONSY];

    float[] fromHiddenYToInputGradients = new float[INPUT_NEURONS];
    float[] fromHiddenXToInputGradients = new float[INPUT_NEURONS];

    // Глобальные переменные
    public static int count = 1;
    public static float initialLearningRate = 0.000012f;
    /**
     * Speed of learning of NN. Can be adapting for optimizing education. Define value is 0.015
     */
    public static float learningRateX = initialLearningRate;
    public static float learningRateY = initialLearningRate;
    /**
     * Path to save wiegths
     */
    static String filename = "C:\\Users\\user\\Desktop\\GazeTracker\\weights.txt";
    public static Random rand = new Random();
    public static boolean isTested = true; 
    public static boolean flagX = true;
    public static boolean flagY = false;
    // public AdamOptimizer optimizerOutputToHiddenX = new AdamOptimizer(HIDDEN_NEURONSX, HIDDEN_NEURONSX, 0.9f, 0.999f, 4e-8f);
    // public AdamOptimizer optimizerHiddenToInputX = new AdamOptimizer(INPUT_NEURONS, INPUT_NEURONS, 0.9f, 0.999f, 4e-8f);
    // public AdamOptimizer optimizerOutputToHiddenY = new AdamOptimizer(HIDDEN_NEURONSX, HIDDEN_NEURONSY, 0.9f, 0.999f, 4e-8f);
    // public AdamOptimizer optimizerHiddenToInputY = new AdamOptimizer(INPUT_NEURONS, INPUT_NEURONS, 0.9f, 0.999f, 4e-8f);
    public static float x;
    public static float y;
    public static float errorX, errorY;
    public static float[] grayImage;
    public static int width = 1080;
    public static int height = 2200;
    public static float lossX;
    public static float lossY;
    public static float newLossX;
    public static float newLossY;
    public int portionInput = INPUT_NEURONS / 20;
    public int portionHidden = HIDDEN_NEURONSX / 20;
    /**
     * lambda for L1 and L2 regularization algorithms. Value is 0.01
     */
    public static float lambda = 0.001f;
    public float lossAverage = 0;
    public float alpha = 0.01f; // коэффициент для скользящего среднего
    public static float beta = 0.01f; // коэффициент для изменения скорости обучения
    public static String DATASET = "C:\\Users\\user\\Desktop\\batch";
    public static String DATASET_1 = "C:\\Users\\user\\Desktop\\dataset1";
    public static String EXECUTE = "C:\\Users\\user\\Desktop\\execute";
    public static int backproploops = 0;
    public static String fileName;

    public static GazeTracker network = new GazeTracker();
    public static SaveAndLoadNN saveAndLoadNN = new SaveAndLoadNN(network);
    
    // Конструктор
    public GazeTracker() {
        // Инициализация весов из файла или случайными значениями
        if(isTested){
            System.out.println("Create a NN...");
        }
    }

    // Инициализация весов
    public void initializeWeights() {
        if(isTested){
            System.out.println("initialize input-hidden X weigths");
        }
        for (int i = 0; i <INPUT_NEURONS; i++) {
            for (int j = 0; j <HIDDEN_NEURONSX; j++) {
                inputToHiddenWeightsX[i][j] = (float) (rand.nextGaussian() * Math.sqrt(2.0 / (INPUT_NEURONS + HIDDEN_NEURONSX)));
            }
        }
        if(isTested){
            System.out.println("initialize input-hidden Y weigths");
        }
        for (int i = 0; i <INPUT_NEURONS; i++) {
            for (int j = 0; j <HIDDEN_NEURONSY; j++) {
                inputToHiddenWeightsY[i][j] = (float) (rand.nextGaussian() * Math.sqrt(2.0 / (INPUT_NEURONS + HIDDEN_NEURONSY)));
            }
        }

        if(isTested){
            System.out.println("initialize hidden-output X weigths");
        }
        for (int i = 0; i <HIDDEN_NEURONSX; i++) {
            for (int j = 0; j <OUTPUT_NEURONSX; j++) {
                hiddenToOutputWeightsX[i][j] = (float) (rand.nextGaussian() * Math.sqrt(2.0 / (HIDDEN_NEURONSX + OUTPUT_NEURONSX)));
            }
        }

        if(isTested){
            System.out.println("initialize hidden-output Y weigths");
        }
        for (int i = 0; i <HIDDEN_NEURONSY; i++) {
            for (int j = 0; j <OUTPUT_NEURONSY; j++) {
                hiddenToOutputWeightsY[i][j] = (float) (rand.nextGaussian() * Math.sqrt(2.0 / (HIDDEN_NEURONSY + OUTPUT_NEURONSY)));
            }
        }
    }

    // Прямое распространение
    public void feedForward(float[] grayImage) {
        
        // Преобразование RGB изображения в одномерный массив
        for (int i = 0; i <grayImage.length; i++) {
            inputLayer[i] = grayImage[i];
        }

        // Вычисление значений скрытого слоя
        for (int i = 0; i < HIDDEN_NEURONSX; i++) {
            hiddenLayerX[i] = 0;
            hiddenLayerY[i] = 0;
            for (int j = 0; j < INPUT_NEURONS; j++) {
                hiddenLayerX[i] += inputLayer[j] * inputToHiddenWeightsX[j][i];
                hiddenLayerY[i] += inputLayer[j] * inputToHiddenWeightsY[j][i];
            }
            hiddenLayerX[i] = activationFunction(hiddenLayerX[i]);
            hiddenLayerY[i] = activationFunction(hiddenLayerY[i]);
        }

        // Вычисление значений выходного слоя
        for (int i = 0; i < OUTPUT_NEURONSX; i++) {
            outputLayerX[i] = 0;
            for (int j = 0; j <HIDDEN_NEURONSX; j++) {
                outputLayerX[i] +=hiddenLayerX[j] *hiddenToOutputWeightsX[j][i];
            }
            outputLayerX[i] = activationFunction(outputLayerX[i]);
        }

        for (int i = 0; i < OUTPUT_NEURONSY; i++) {
            outputLayerY[i] = 0;
            for (int j = 0; j < HIDDEN_NEURONSY; j++) {
                outputLayerY[i] += hiddenLayerY[j] * hiddenToOutputWeightsY[j][i];
            }
            outputLayerY[i] = activationFunction(outputLayerY[i]);
        }

    }

    // Функция активации (например, сигмоид)
    private static float activationFunction(float x) {
        return (float) (1 / (1 + Math.exp(-x)));
    }

    // Геттеры для выходных значений
    public  float getXCoordinate(int width) {
        return (outputLayerX[0] * width);
    }

    public float getYCoordinate(int heigth) {
        return (outputLayerY[0] * heigth);
    }

    // Основная функция для тестирования
    public static void main(String[] args) throws IOException {

        saveAndLoadNN.loadWeights(filename);

        if(isTested){
            System.out.println("Hello, I'm in main method...");
        }

        // Загрузка DataFiles
        // Просто продолжаем обучать в том же направлении, ничего не меняя
        File folder = new File(DATASET);
        File[] listOfFiles = folder.listFiles();

        if(isTested){
            System.out.println("I start to prepare data, and learn NN...");
        }

        for (File file : listOfFiles) {
            
            if (file.isFile()) {
                // Тестовое RGB изображение (здесь должен быть ваш одномерныйefh,../;
                BufferedImage originalImage = ImageIO.read(file);
                grayImage = ImageProcessor.convertToNormalizedArray(originalImage, 320, 480);

                fileName = file.getName();
                fileName = fileName.replace("files_", ""); // Удаление "files_"
                String[] splitName = fileName.split("_");
                x = Float.parseFloat(splitName[0]);
                y = Float.parseFloat(splitName[1].split("\\.")[0] + "." + splitName[1].split("\\.")[1]);
                
                if(isTested & count % 10 == 0){ 
                    System.out.println("Start to learn of NN " + count + " times.");
                }

                network.feedForward(grayImage);
                network.backpropagation(grayImage, x, y);
                
                if(isTested & count % 400 == 0){
                    System.out.println("Save weights to file after " + count + " learning iterations...");
                    saveAndLoadNN.saveWeights(filename, network);
                    System.out.println("Save is succesfull!");
                }
                count++;
                
            }
        }
        saveAndLoadNN.saveWeights(filename, network);

    }

    // Метод обратного распространения ошибки
    public void backpropagation(float[] grayImage, float expectedX, float expectedY) {

        // tempWeightsSaver();
        float predictedX = outputLayerX[0];
        float predictedY = outputLayerY[0];
        float actualX = expectedX / 1080;
        float actualY = expectedY / 2200;
        lossX = mseLossX(predictedX, actualX);
        lossY = mseLossY(predictedY, actualY);
        if(isTested & count % 10 == 0){
            printTestResults("value before backpropogagion: ", lossX, lossY);
        }
       
        // Вычисление ошибки для выходного слоя (предполагаем, что у нас 2 выходных нейрона)
        errorX = outputLayerX[0] - expectedX / 1080;
        errorY = outputLayerY[0] - expectedY / 2200;

        // Градиенты для выходного слоя
        outputGradientsX[0] = errorX * derivativeActivationFunction(outputLayerX[0]);
        outputGradientsY[0] = errorY * derivativeActivationFunction(outputLayerY[0]);
        

        // Градиенты для скрытого слоя X
        for (int i = 0; i < HIDDEN_NEURONSX; i++) {
            hiddenGradientsX[i] = 0;
            for (int j = 0; j < OUTPUT_NEURONSX; j++) {
                hiddenGradientsX[i] += outputGradientsX[j] * hiddenToOutputWeightsX[i][j];
            }
            hiddenGradientsX[i] *= derivativeActivationFunction(hiddenLayerX[i]);
        }
        
        // Градиенты для скрытого слоя Y
        for (int i = 0; i < HIDDEN_NEURONSY; i++) {
            hiddenGradientsY[i] = 0;
            for (int j = 0; j < OUTPUT_NEURONSY; j++) {
                hiddenGradientsY[i] += outputGradientsY[j] * hiddenToOutputWeightsY[i][j];
            }
            hiddenGradientsY[i] *= derivativeActivationFunction(hiddenLayerY[i]);
        }

        // Градиенты от скрытого слоя X до входного слоя
        for (int i = 0; i < INPUT_NEURONS; i++){
            fromHiddenXToInputGradients[i] = 0;
            for (int j = 0; j < HIDDEN_NEURONSX; j++){
                fromHiddenXToInputGradients[i] += hiddenGradientsX[j] * inputToHiddenWeightsX[i][j];
            }
            fromHiddenXToInputGradients[i] *= derivativeActivationFunction(inputLayer[i]);
        }

        // Градиенты от скрытого слоя Y до входного слоя
        for (int i = 0; i < INPUT_NEURONS; i++){
            fromHiddenYToInputGradients[i] = 0;
            for (int j = 0; j < HIDDEN_NEURONSY; j++){
                fromHiddenYToInputGradients[i] += hiddenGradientsY[j] * inputToHiddenWeightsY[i][j];
            }
            fromHiddenYToInputGradients[i] *= derivativeActivationFunction(inputLayer[i]);
        }

        GetLearningRateSpeedX(predictedX, actualX, lossX);
        GetLearningRateSpeedY(predictedY, actualY, lossY);

        // Обновление весов для скрытого к выходному слою Х
        flagX = true;
        flagY = false;
        new AdamOptimizer(HIDDEN_NEURONSX, HIDDEN_NEURONSX, 0.9f, 0.999f, 4e-8f).updateOutputToHidden(hiddenToOutputWeightsX, hiddenGradientsX);

        // Обновление весов для скрытого к выходному слою Y
        flagX = false;
        flagY = true;
        new AdamOptimizer(HIDDEN_NEURONSX, HIDDEN_NEURONSY, 0.9f, 0.999f, 4e-8f).updateOutputToHidden(hiddenToOutputWeightsY, hiddenGradientsY);

        // Обновление весов для входного к скрытому слою X
        flagX = true;
        flagY = false;
        // new AdamOptimizer(INPUT_NEURONS, INPUT_NEURONS, 0.9f, 0.999f, 4e-8f).updateHiddenToInput(inputToHiddenWeightsX, fromHiddenXToInputGradients);
        portionalWeightsOptimizer(portionInput, portionHidden, inputToHiddenWeightsX, fromHiddenXToInputGradients);

        // Обновление весов для входного к скрытому слою Y
        flagX = false;
        flagY = true;
        // new AdamOptimizer(INPUT_NEURONS, INPUT_NEURONS, 0.9f, 0.999f, 4e-8f).updateHiddenToInput(inputToHiddenWeightsY, fromHiddenYToInputGradients);
        portionalWeightsOptimizer(portionInput, portionHidden, inputToHiddenWeightsY, fromHiddenYToInputGradients);

        network.feedForward(grayImage);
        float newPredictedX = outputLayerX[0];
        float newPredictedY = outputLayerY[0];
        GetLearningRateSpeedX(newPredictedX, actualX, lossX);
        GetLearningRateSpeedY(newPredictedY, actualY, lossY);
        
        if(isTested & count % 10 == 0){
            newLossX = mseLossX(newPredictedX, actualX);
            newLossY = mseLossY(newPredictedY, actualY);
            printTestResults("Values after backpropogation: ", newLossX, newLossY);
        }

        // if(newLossX > lossX || newLossY > lossY){

        //     recoverTempWeights();
        //     Path sourcePath = new File(DATASET, "files_" + GazeTracker.fileName).toPath();
        //     Path targetPath = new File(EXECUTE, "files_" + GazeTracker.fileName).toPath();

        //     try {
        //         Files.move(sourcePath, targetPath, StandardCopyOption.REPLACE_EXISTING);
        //     } catch (IOException e) {
        //         e.printStackTrace();
        //     }
        // }
        

    }

    private void portionalWeightsOptimizer(int portionInput, int portionHidden, float[][] inputToHiddenWeights, float[] fromHiddenToInputGradients){
        
        float[][] tempWeights = new float[portionInput][portionHidden];
        float[] tempGradients = new float[portionInput];

        int weightsStart = portionInput;
        int gradientsStart = portionInput;
        int weightsStart_1 = portionHidden;
        int weigthsStep_1 = weightsStart_1;
        int weigthsStep = weightsStart;
        int gradientsStep = gradientsStart;

        for (int i = 0; i < 20; i++){
            int start = 0;
            for (int j = weightsStart - weigthsStep; j < weightsStart; j++){
                int end = 0;
                for (int k = weightsStart_1 - weigthsStep_1; k < weightsStart_1; k++){
                    tempWeights[start][end] = inputToHiddenWeights[j][k];
                    end++;
                }
                start++;
            }

            start = 0;
            for (int l = gradientsStart - gradientsStep; l < gradientsStart; l++){
                // проработать индексы для tempGradients потому что будет выход за пределы массива.
                tempGradients[start++] = fromHiddenToInputGradients[l];
            }

            new AdamOptimizer(portionInput, portionHidden, 0.9f, 0.999f, 4e-8f).updateHiddenToInput(tempWeights, tempGradients);

            start = 0;
            for (int j = weightsStart - weigthsStep; j < weightsStart; j++){
                int end = 0;
                for (int k = weightsStart_1 - weigthsStep_1; k < weightsStart_1; k++){
                    inputToHiddenWeights[j][k] = tempWeights[start][end];
                    end++;
                }
                start++;
            }

            start = 0;
            for (int l = gradientsStart - gradientsStep; l < gradientsStart; l++){
                fromHiddenToInputGradients[l] = tempGradients[start++];
            }

            weightsStart += weigthsStep;
            gradientsStart += gradientsStep;
            weightsStart_1 += weigthsStep_1;
        }
        
    }

    private void tempWeightsSaver(){
        
        tempOutputX[0] = outputLayerX[0];
        tempOutputY[0] = outputLayerY[0];

        for(int i = 0; i < HIDDEN_NEURONSX; i++){
            for(int j = 0; j < OUTPUT_NEURONSX; j++){
                tempHiddenToOutputX[i][j] = hiddenToOutputWeightsX[i][j];
            }
        }

        for(int i = 0; i < HIDDEN_NEURONSY; i++){
            for(int j = 0; j < OUTPUT_NEURONSY; j++){
                tempHiddenToOutputY[i][j] = hiddenToOutputWeightsY[i][j];
            }
        }

        for (int i = 0; i < INPUT_NEURONS; i++){
            for (int j = 0; j < HIDDEN_NEURONSX; j++){
                tempInputToHiddenXWeights[i][j] = inputToHiddenWeightsX[i][j];
            }
        }

        for (int i = 0; i < INPUT_NEURONS; i++){
            for (int j = 0; j < HIDDEN_NEURONSY; j++){
                tempInputToHiddenYWeights[i][j] = inputToHiddenWeightsY[i][j];
            }
        }

    }

    private void recoverTempWeights(){

        outputLayerX[0] = tempOutputX[0];
        outputLayerY[0] = tempOutputY[0];

        for(int i = 0; i < HIDDEN_NEURONSX; i++){
            for(int j = 0; j < OUTPUT_NEURONSX; j++){
                hiddenToOutputWeightsX[i][j] = tempHiddenToOutputX[i][j];
            }
        }

        for(int i = 0; i < HIDDEN_NEURONSY; i++){
            for(int j = 0; j < OUTPUT_NEURONSY; j++){
                hiddenToOutputWeightsY[i][j] = tempHiddenToOutputY[i][j];
            }
        }

        for (int i = 0; i < INPUT_NEURONS; i++){
            for (int j = 0; j < HIDDEN_NEURONSX; j++){
                inputToHiddenWeightsX[i][j] = tempInputToHiddenXWeights[i][j];
            }
        }

        for (int i = 0; i < INPUT_NEURONS; i++){
            for (int j = 0; j < HIDDEN_NEURONSY; j++){
                inputToHiddenWeightsY[i][j] = tempInputToHiddenYWeights[i][j];
            }
        }
    }


    public static float mseLossX(float predicted, float actual) {
        return (float)Math.pow(predicted - actual, 2);
    }

    public static float mseLossY(float predicted, float actual) {
        return (float)Math.pow(predicted - actual, 2);
    }

    // Производная функции активации (например, сигмоид)
    private static float derivativeActivationFunction(float x) {
        float sigmoid = activationFunction(x);
        return sigmoid * (1 - sigmoid);
    }

    // L1 регуляризация
    public static float l1Regularization(float weight) {
        return lambda * Math.signum(weight);
    }

    // L2 регуляризация
    public static float l2Regularization(float weight) {
        return lambda * weight;
    }

    public static void GetLearningRateSpeedX(float predictedX, float actualX, float lossX){

        if (lossX <= (float)mseLossX(predictedX, actualX)) {
            // Увеличиваем скорость обучения только за счет значения функции потерь
            learningRateX += beta * learningRateX;
        } else if(lossX <= 0.001 & learningRateX - lossX > 0) {
            // Уменьшаем скорость обучения
            learningRateX -= lossX;
        }else{
            // Возвращаем скорость обучения к изначальному значению-
            learningRateX = initialLearningRate;
        }
    }

    public static void GetLearningRateSpeedY(float predictedY, float actualY, float lossY){

        if (lossY <= (float)mseLossY(predictedY, actualY)) {
            // Увеличиваем скорость обучения только за счет значения функции потерь
            learningRateY += beta * learningRateY;
        } else if (lossY <= 0.001 & learningRateY - lossY > 0){
            // Уменьшаем скорость обучения
            learningRateY -= lossY;
        }else{
            // Возвращаем скорость обучения к изначальному значению
            learningRateY = initialLearningRate;
        }
    }

    private void printTestResults(String message, float lossX, float lossY) {
        if(isTested){
            System.out.println(message);
            System.out.println(" The value of outputLayerX is " + outputLayerX[0]);
            System.out.println(" The value of outputLayerY is " + outputLayerY[0]);
            System.out.println("NN says that x is " + network.getXCoordinate(SCREEN_WIDTH) + ", but the x is " + x);
            System.out.println("NN says that y is " + network.getYCoordinate(SCREEN_HEIGTH) + ", but the y is " + y);
            System.out.println("Learning Rate for X is: " + learningRateX + ", and Loss for X is: " + lossX);
            System.out.println("Learning Rate for Y is: " + learningRateY + ", and Loss for Y is: " + lossY);
        }
    }
}