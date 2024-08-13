import java.awt.Dimension;
import java.awt.Toolkit;

public final class ScreenSizeGetter {

    // Геттер для ширины экрана
    public static int getScreenWidth() {
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        return screenSize.width;
    }

    // Геттер для высоты экрана
    public static int getScreenHeight() {
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        return screenSize.height;
    }
}

