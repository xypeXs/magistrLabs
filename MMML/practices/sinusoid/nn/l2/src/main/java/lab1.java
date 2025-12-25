import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class lab1 {
    public static void main(String[] args){

        try(FileReader fr = new FileReader("src/Main/Tests.txt")){
            System.out.print("\u001b[35m");
            BufferedReader reader = new BufferedReader(fr);
            ArrayList<Test> allTest = new ArrayList<>();
            int inEq = 0;
            int outEq = 0;

            String line = reader.readLine();
            int testNum = Integer.parseInt(line);
//            System.out.println(testNum);

            for (int tests=0; tests < testNum; tests++){
                ArrayList<Float> testIn = new ArrayList<>();
                ArrayList<Float> testOut = new ArrayList<>();

                line = reader.readLine();
                String[] ln = line.split(" ");
//                System.out.println(line);
                inEq = ln.length;

                for (int in=0; in < ln.length; in++){
                    testIn.add(Float.parseFloat(ln[in]));
                }

                line = reader.readLine();
                ln = line.split(" ");
//                System.out.println(line);
                outEq = ln.length;

                for (int in=0; in < ln.length; in++){
                    testOut.add(Float.parseFloat(ln[in]));
                }

                Test test = new Test();
                test.initTest(testIn, testOut);
                allTest.add(test);
            }

            System.out.println();
            System.out.println("Число входов " + inEq);
            System.out.println("Число нейронов " + outEq);
            System.out.println();

            ArrayList<Neuron> neurons = new ArrayList<>();
            for (int neur = 0; neur < outEq; neur++){
                Neuron neuron = new Neuron();
                neuron.setTriger(-1.0F);//смещение
                neuron.initNeuron(inEq);
//                System.out.println("Начальные веса " + (neur+1) + " нейрнона: ");
//                System.out.println(neuron.getListOfWeight());
//                System.out.println();
                neurons.add(neuron);
            }
//            System.out.println();

            int E = 1;
            int era = 0;

            float lim = 0.0F;  //граница ошибки
            float alpha = 0.01F; //альфа
            int countInTest = 1;
            int coutnAll = 1;
            int step = 1; //шаг статистики

            ArrayList<Integer> stats = new ArrayList<>();
            while ((E>lim) & (coutnAll<testNum)) {
                era++;
                E = 0;
                float sum = 0.0F;
                coutnAll = 0;

                for (int curTest = 0; curTest < testNum; curTest++){    //Итерация
//                    System.out.println("|||||||||||||||||||||||||");
//                    System.out.print((curTest+1) + " тест ");
                    Test test = allTest.get(curTest);
                    ArrayList<Float> testIn = test.getTestIn();
                    ArrayList<Float> testOut = test.getTestOut();
                    countInTest = 0;

                    for (int curNeuron = 0; curNeuron < outEq; curNeuron++){
//                        System.out.println((curNeuron+1) + " нейрон ");
                        float result = neurons.get(curNeuron).calculatingLerning(testIn, testOut.get(curNeuron));
                        float e = testOut.get(curNeuron)-result;
                        if (neurons.get(curNeuron).changeWeight(alpha, testIn, e))
                            countInTest++;
                        sum += (testOut.get(curNeuron)-result) * (testOut.get(curNeuron)-result);
//                        System.out.print(sum + " ");
//                        System.out.println(testOut.get(curNeuron) + " " + result);
                    }
//                    System.out.println("Связей не изменилось в тесте " + (curTest+1) + " - " + countInTest);
                    if (countInTest == inEq) coutnAll++;
                }

                E = Math.round((1/testNum) + sum);

//                System.out.println("Ошибка сети " + E);
//                System.out.println("Эра " + era + " Ошибка сети " + E);
//                System.out.println(sum);
//                System.out.println();

                if (era % step == 0) {
                    System.out.println("Эра " + era + " Ошибка сети " + E + " Текущая альфа: "+ alpha);
                    System.out.println("Нейронов не изменилось в последнем тесте " + countInTest);
                    for (int curNeuron = 0; curNeuron < outEq; curNeuron++){
                        System.out.println(neurons.get(curNeuron).getListOfWeight());
                    }
                    System.out.println();
                    stats.add(E);
//                    break;
                }
            }

            System.out.println();
            System.out.println("|||||||||||||||||||||||||||||||||");
            if (countInTest == outEq){
                System.out.println("Вектор весов перестал изменяться");
            }
            System.out.println("Последнее значение ошибки: " + E);
            System.out.println("Последняя эпоха: " + era);
            System.out.println("Значение ошибки каждые " + step + " эпохи: ");
            System.out.println(stats);

            System.out.println("Тестирование обучения: ");
            while (true){
                System.out.println();
                System.out.println("Введите тест: ");

                ArrayList<Float> testing = new ArrayList<>();
                Scanner console = new Scanner(System.in);
                String lineTest = console.nextLine();
                String[] lnTest = lineTest.split(" ");
                for (int a = 0; a < inEq; a++){
                    testing.add(Float.parseFloat(lnTest[a]));
                }

                for (int curNeuron = 0; curNeuron < outEq; curNeuron++){
//                    System.out.println((curNeuron+1) + " нейрон ");
//                    System.out.print(Math.round(neurons.get(curNeuron).calculating(testing)) + " ");
                    if (neurons.get(curNeuron).calculating(testing) > 0) System.out.print(curNeuron + " ");
                }
            }

        }
        catch(IOException ex){
            System.out.println(ex.getMessage());
        }
    }
}
