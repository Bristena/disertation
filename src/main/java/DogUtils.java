import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

public class DogUtils {
    private Map<String, INDArray> partMap;
    private INDArray array;

    public Map<String, INDArray> getPartMap() {
        return partMap;
    }

    public void setPartMap(Map<String, INDArray> partMap) {
        this.partMap = partMap;
    }

    public INDArray getArray() {
        return array;
    }

    public void setArray(INDArray array) {
        this.array = array;
    }
}
