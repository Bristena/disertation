import org.nd4j.linalg.api.ndarray.INDArray;

public class Box {
    private INDArray boxCorners;
    private INDArray eyeSlope;
    private INDArray interEyeDist;

    public INDArray getBoxCorners() {
        return boxCorners;
    }

    public void setBoxCorners(INDArray boxCorners) {
        this.boxCorners = boxCorners;
    }

    public INDArray getEyeSlope() {
        return eyeSlope;
    }

    public void setEyeSlope(INDArray eyeSlope) {
        this.eyeSlope = eyeSlope;
    }

    public INDArray getInterEyeDist() {
        return interEyeDist;
    }

    public void setInterEyeDist(INDArray interEyeDist) {
        this.interEyeDist = interEyeDist;
    }
}
