import java.util.ArrayList;

public class Test {
    private ArrayList<Float> testIn;
    private ArrayList<Float> testOut;

    public void initTest(ArrayList<Float> in, ArrayList<Float> out) {
        testIn = in;
        testOut = out;
    }

    public ArrayList<Float> getTestIn() {
        return testIn;
    }

    public ArrayList<Float> getTestOut() {
        return testOut;
    }
}
