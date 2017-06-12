import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TrainData {

    private Integer a[] = {0, 1, 2, 3};
    private Integer b[] = {4, 5};
    private Integer c[] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    private List<List<Integer>> MODEL_MASKS = new ArrayList<>();

    public void run() {
        MODEL_MASKS.add(Arrays.asList(a));
        MODEL_MASKS.add(Arrays.asList(b));
        MODEL_MASKS.add(Arrays.asList(c));
    }

}
