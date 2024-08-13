import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;

public class ImageProcessor {

    public static float[] convertToNormalizedArray(BufferedImage image, int targetWidth, int targetHeight) {
        // Изменение размера изображения
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resizedImage.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2d.drawImage(image, 0, 0, targetWidth, targetHeight, null);
        g2d.dispose();
    
        // Создание одномерного массива для нормализованных значений RGB
        float[] normalizedArray = new float[targetWidth * targetHeight];
    
        // Преобразование и нормализация значений в серый
        for (int i = 0; i < targetHeight; i++) {
            for (int j = 0; j < targetWidth; j++) {
                int rgb = resizedImage.getRGB(j, i);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                float grayLevel = (r + g + b) / 3;

                normalizedArray[i * targetWidth + j] = (float) (grayLevel / 255.0);
            }
        }
    
        return normalizedArray;
    }
}
