import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class Main {
    public static void pngInTest() {
        picture pic = new picture();
        File folder = new File("src/Main/img");
        File[] listOfFiles = folder.listFiles();

        assert listOfFiles != null;
        for (File file : listOfFiles) {
            if (file.isFile()) {
                String way = pic.colInBw(file.getAbsolutePath());
                if (way.equals("")) {
                    pic.pngToTxt(way);
                }
            }
            System.out.println(file.getAbsolutePath());
        }
    }

    public static void main(String[] args) {
        pngInTest();

        ArrayList<String> test = new ArrayList<>();
        File folder = new File("src/Main/txt");
        File[] listOfFiles = folder.listFiles();

        assert listOfFiles != null;
        for (File file : listOfFiles) {
            if (file.isFile()) {
                try {
                    BufferedReader reader = new BufferedReader(new FileReader(file));
                    test.add(reader.readLine());
                } catch (IOException e) {
                    System.out.println("Файл не найден или не удалось сохранить");
                }
            }
        }

        int n = test.get(0).split(" ").length;
        int[][] x = new int[test.size()][n];

        for (int i = 0; i < test.size(); i++) {
            String[] testSplit = test.get(i).split(" ");
            for (int j = 0; j < n; j++) {
                x[i][j] = Integer.parseInt(testSplit[j]);
            }
        }

        Float[][] weight = new Float[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                weight[i][j] = 0F;
            }
        }

        for (int testCount = 0; testCount < test.size(); testCount++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    weight[i][j] += x[testCount][i] * x[testCount][j];
                    //System.out.print(weight[i][j] + " ");
                }
                //System.out.println();
            }
            //System.out.println();
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                weight[i][j] *= (float) 1 / n;
                if (i == j) weight[i][j] = 0F;
            }
        }

        /*
        for (int i=0; i<n; i++){
            for (int j=0; j<n; j++){
                System.out.print(weight[i][j] + " ");
            }
            System.out.println();
        }
        */

        System.out.println("test");
        Scanner in = new Scanner(System.in);
//        System.out.println("Размеры изображения");

//        int sizex = in.nextInt();
//        int sizey = in.nextInt();
        System.out.println("Тестирование");


        while (true) {
//          Ввод и вывод по картинку
            System.out.println("Введите имя файла для теста:");
            String name = in.nextLine();
            picture pic = new picture();
            int[] testIn = pic.inTest(name);
            float[] testOut = new float[n];
            for (int i = 0; i < n; i++) {
                testOut[i] = 0;
            }

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    testOut[i] += weight[i][j] * testIn[j];
                }
                if (testOut[i] >= 0) testOut[i] = 1;
                else testOut[i] = -1;
            }

            pic.outTest(testOut);

//            Ввод и вывод консоли
//            int[] check = new int[n];
//            float[] y = new float[n];
//
//            for (int i=0; i<n; i++){
//                check[i] = in.nextInt();
//            }
//            for (int i=0; i<n; i++){
//                y[i] = 0;
//            }
//
//            for (int i=0; i<n; i++){
//                for (int j=0; j<n; j++) {
//                    y[i] += weight[i][j] * check[j];
//                }
//                if (y[i] >= 0) y[i] = 1;
//                    else y[i] = -1;
//            }
//
//            int count = 0;
//            for (int i=0; i<sizey; i++) {
//                for (int j = 0; j < sizex; j++) {
//                    if (y[count]>0) System.out.print("0");
//                    else System.out.print("_");
//                    count++;
//                }
//                System.out.println();
//            }
        }
    }
}
