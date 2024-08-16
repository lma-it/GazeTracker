public class AdamOptimizer {
    private float beta1;
    private float beta2;
    private float epsilon;
    private float[][] m;
    private float[][] v;
    

    public AdamOptimizer(int numRows, int numCols, float beta1, float beta2, float epsilon) {
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.m = new float[numRows][numCols];
        this.v = new float[numRows][numCols];
    }


    public void updateHiddenToInput(float[][] params, float[] grads) {
        for (int i = 0; i < params.length; i++) {
            for (int j = 0; j < params[i].length; j++) {
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * grads[j];
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * grads[j] * grads[j];
                float mHat = m[i][j] / (1 - beta1);
                float vHat = v[i][j] / (1 - beta2);
                
                if(GazeTracker.flagX){
                    //params[i][j] -= GazeTracker.learningRateX * mHat / (Math.sqrt(vHat) + epsilon);
                    float regularizedGrad = grads[j] + GazeTracker.l1Regularization(params[i][j]) + GazeTracker.l2Regularization(params[i][j]);
                    params[i][j] -= GazeTracker.learningRateX * mHat / (Math.sqrt(vHat) + epsilon) + regularizedGrad;
                }else if(GazeTracker.flagY){
                    //params[i][j] -= GazeTracker.learningRateX * mHat / (Math.sqrt(vHat) + epsilon);
                    float regularizedGrad = grads[j] + GazeTracker.l1Regularization(params[i][j]) + GazeTracker.l2Regularization(params[i][j]);
                    params[i][j] -= GazeTracker.learningRateY * mHat / (Math.sqrt(vHat) + epsilon) + regularizedGrad;
                }
            }
        }
    }

    public void updateOutputToHidden(float[][] params, float[] grads) {
        for (int i = 0; i < params.length; i++) {
            for (int j = 0; j < params[i].length; j++) {
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * grads[j];
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * grads[j] * grads[j];
                
                float mHat = m[i][j] / (1 - beta1);
                float vHat = v[i][j] / (1 - beta2);
                
                if(GazeTracker.flagX){
                    //params[i][j] -= GazeTracker.learningRateX * mHat / (Math.sqrt(vHat) + epsilon);
                    // params[i][j] -= GazeTracker.learningRateX * mHat / (Math.sqrt(vHat) + epsilon) + GazeTracker.l1Regularization(params[i][j]) + GazeTracker.l2Regularization(params[i][j]);
                    float regularizedGrad = grads[j] + GazeTracker.l1Regularization(params[i][j]) + GazeTracker.l2Regularization(params[i][j]);
                    params[i][j] -= GazeTracker.learningRateX * mHat / (Math.sqrt(vHat) + epsilon) + regularizedGrad;
                }else if(GazeTracker.flagY){
                    //params[i][j] -= GazeTracker.learningRateX * mHat / (Math.sqrt(vHat) + epsilon);
                    // params[i][j] -= GazeTracker.learningRateY * mHat / (Math.sqrt(vHat) + epsilon) + GazeTracker.l1Regularization(params[i][j]) + GazeTracker.l2Regularization(params[i][j]);
                    float regularizedGrad = grads[j] + GazeTracker.l1Regularization(params[i][j]) + GazeTracker.l2Regularization(params[i][j]);
                    params[i][j] -= GazeTracker.learningRateY * mHat / (Math.sqrt(vHat) + epsilon) + regularizedGrad;
                }
            }
        }
    }
}

