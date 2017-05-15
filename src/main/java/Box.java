import org.nd4j.linalg.api.ndarray.INDArray;

public class Box {
    private INDArray boxCorners;
    private INDArray eyeSlope;
    private double interEyeDist;

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

    public double getInterEyeDist() {
        return interEyeDist;
    }

    public void setInterEyeDist(double interEyeDist) {
        this.interEyeDist = interEyeDist;
    }
}
