import edu.rsatu.encryption.MainWindow;

import javax.swing.*;

public class Main {

    public static void main(String[] args) {
        JFrame jFrame = getDefaultJFrame("Лаба");
        new MainWindow(jFrame);
    }

    protected static JFrame getDefaultJFrame(String windowName) {
        JFrame jFrame = new JFrame(windowName);
        jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jFrame.pack();
        jFrame.setSize(600, 400);

        return jFrame;
    }
}
