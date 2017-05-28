package model;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Data {

    private INDArray x;
    private INDArray y;

    public INDArray getX() {
        return x;
    }

    public void setX(INDArray x) {
        this.x = x;
    }

    public INDArray getY() {
        return y;
    }

    public void setY(INDArray y) {
        this.y = y;
    }
}
