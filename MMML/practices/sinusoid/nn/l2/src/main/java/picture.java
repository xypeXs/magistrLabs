import javax.imageio.ImageIO;
import java.awt.*;
import java.io.File;
import java.awt.image.BufferedImage;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class picture {
    private int sizeX = 0;
    private int sizeY = 0;
    private String nameTest = "";
    private int type = 0;

    String colInBw(String wayPic) {
        try {
            File file = new File(wayPic);
            BufferedImage source = ImageIO.read(file);

            // Создаем новое пустое изображение, такого же размера
            BufferedImage result = new BufferedImage(source.getWidth(), source.getHeight(), source.getType());

            // Делаем двойной цикл, чтобы обработать каждый пиксель
            for (int x = 0; x < source.getWidth(); x++) {
                for (int y = 0; y < source.getHeight(); y++) {

                    // Получаем цвет текущего пикселя
                    Color color = new Color(source.getRGB(x, y));

                    // Получаем каналы этого цвета
                    int blue = color.getBlue();
                    int red = color.getRed();
                    int green = color.getGreen();

                    // Применяем стандартный алгоритм для получения черно-белого изображения
                    int grey = (int) (red * 0.299 + green * 0.587 + blue * 0.114);

                    int grey1;
                    if (grey < 200) {
                        grey1 = 0;
                    } else grey1 = 255;

                    //  Cоздаем новый цвет
                    Color newColor = new Color(grey1, grey1, grey1);

                    // И устанавливаем этот цвет в текущий пиксель результирующего изображения
                    result.setRGB(x, y, newColor.getRGB());
                }
            }

            // Созраняем результат в новый файл
            String[] worlds = wayPic.split("img");
            String out = worlds[0] + "bw" + worlds[1];
            File output = new File(out);
            ImageIO.write(result, "png", output);
            return out;

        } catch (IOException e) {
            System.out.println("Файл не найден или не удалось сохранить");
        }
        return "";
    }


    void pngToTxt(String wayPic) {
        try {
            File file = new File(wayPic);
            BufferedImage source = ImageIO.read(file);
            String[] way = wayPic.split("bw");
            FileWriter file1 = new FileWriter(way[0] + "txt" + way[1].split(".png")[0] + ".txt");
            for (int x = 0; x < source.getHeight(); x++) {
                for (int y = 0; y < source.getWidth(); y++) {

                    // Получаем цвет текущего пикселя
                    Color color = new Color(source.getRGB(y, x));
                    int rgb = color.getRGB();
                    if (rgb < -1) {
                        file1.write("1 ");
                    } else file1.write("-1 ");

                }
            }
            file1.close();

        } catch (IOException e) {
            System.out.println("Файл не найден или не удалось сохранить");
        }
    }

    int[] inTest(String name) {
        try {
            this.nameTest = "src/main/test/" + name;
            File file = new File(this.nameTest);
            BufferedImage source = ImageIO.read(file);
            this.sizeX = source.getWidth();
            this.sizeY = source.getHeight();
            this.type = source.getType();
            int[] res = new int[this.sizeX * this.sizeY];
            int count = 0;

            // Делаем двойной цикл, чтобы обработать каждый пиксель
            for (int y = 0; y < source.getHeight(); y++) {
                for (int x = 0; x < source.getWidth(); x++) {

                    Color color = new Color(source.getRGB(x, y));

                    int blue = color.getBlue();
                    int red = color.getRed();
                    int green = color.getGreen();

                    int grey = (int) (red * 0.299 + green * 0.587 + blue * 0.114);

                    if (grey < 200) {
                        res[count] = 1;
                    } else {
                        res[count] = -1;
                    }
                    count++;
                }
            }
            System.out.println();
            return res;

        } catch (IOException e) {
            System.out.println("Файл не найден или не удалось сохранить");
        }
        int[] ex = new int[0];
        return ex;
    }

    void outTest(float[] out) {
        try {
            // Создаем новое пустое изображение, такого же размера
            BufferedImage result = new BufferedImage(this.sizeX, this.sizeY, this.type);
            int count = 0;
            int[] out1 = new int[this.sizeX * this.sizeY];

            // Делаем двойной цикл, чтобы обработать каждый пиксель
            for (int y = 0; y < this.sizeY; y++) {
                for (int x = 0; x < this.sizeX; x++) {
//                    if (out[count] == 1) System.out.print("0");
//                    else System.out.print("_");

                    if (out[count] > 0) {
                        out1[count] = 0;
                    } else out1[count] = 255;
                    Color newColor = new Color(out1[count], out1[count], out1[count]);

                    // И устанавливаем этот цвет в текущий пиксель результирующего изображения
                    result.setRGB(x, y, newColor.getRGB());
                    count++;
                }
            }
            // Созраняем результат в новый файл
            File output = new File(this.nameTest.split(".png")[0] + "OUT.png");
            System.out.println("Out in " + this.nameTest.split(".png")[0] + "OUT.png");
            ImageIO.write(result, "png", output);
            Desktop.getDesktop().open(new File(output.getAbsolutePath()));

        } catch (IOException e) {
            System.out.println("Файл не найден или не удалось сохранить");
        }
    }
}
