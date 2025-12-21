import java.util.ArrayList;

import static java.lang.Math.exp;

public class Neuron {
    private ArrayList<Float> listOfWeight;
    private float triger = 0.0F;

    public void initNeuron(int x){
        listOfWeight = new ArrayList<>();
        for (int i=0; i < x; i++){
            float RandomWeight = (float) Math.random();
            listOfWeight.add(RandomWeight);
        }
    }

    public float getTriger() {
        return triger;
    }

    public void setTriger(float triger) {
        this.triger = triger;
    }

    public ArrayList<Float> getListOfWeight(){
        return listOfWeight;
    }

    private void setListOfWeight(ArrayList<Float> listOfWeightIn) {
        this.listOfWeight = listOfWeightIn;
    }

    private float stepwize (float value){
        if (value > (0 + triger))
            return 1.0F;
            else return 0.0F;
    }

    private float sigmod (float value){
        if (value > (1 / (1 + exp(-1)) + triger))
            return 1.0F;
            else return 0.0F;
    }

    public float calculatingLerning (ArrayList<Float> inputTests, float outTest) {
        float result = 0.0F;
        ArrayList<Float> weight = getListOfWeight();
//        System.out.print("----Вход ");
        for (int curS = 0; curS < inputTests.size(); curS++){
            result += inputTests.get(curS) * weight.get(curS);
//            System.out.print(inputTests.get(curS) + " ");
        }
//        System.out.println();
//        System.out.println("----Ответ " + result);

//        System.out.println("----Ответ " + stepwize(result));
//        return stepwize(result);

//        System.out.println("----Ответ " + sigmod(result));
        return sigmod(result);

//        System.out.println("------Ожидалось " + outTest);
//        System.out.println("--Ошибка " + (result - outTest));
//        System.out.println();
//        return result;
    }

    public float calculating (ArrayList<Float> inputTests) {
        float result = 0.0F;
        ArrayList<Float> weight = getListOfWeight();
        for (int curS = 0; curS < inputTests.size(); curS++) {
            result += inputTests.get(curS) * weight.get(curS);
        }
//        System.out.print(stepwize(result) + " ");
//        System.out.print(sigmod(result) + " ");
//        System.out.println(result);
        return sigmod(result);
    }


    public boolean changeWeight(float alpha, ArrayList<Float> inputTests, float e){
        int count = 0;
        ArrayList<Float> weightOld = getListOfWeight();
        ArrayList<Float> weightNew = new ArrayList<>();
        for (int i = 0; i < inputTests.size(); i++){
            float delta = (alpha * inputTests.get(i) * e);
            float weight = weightOld.get(i) + delta;
//            System.out.println(weightOld.get(i) + " + " + delta + " = " + weight);
//            System.out.print(delta + " ");
            weightNew.add(weight);
        }
        setListOfWeight(weightNew);
        return weightNew.equals(weightOld);
//        System.out.println(weightOld);
//        System.out.println(weightNew);
    }
}
