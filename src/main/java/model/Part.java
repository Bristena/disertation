package model;

public class Part {
    private String part;
    private int position;

    public Part(String part, int position) {
        this.part = part;
        this.position = position;
    }

    public String getPart() {
        return part;
    }

    public void setPart(String part) {
        this.part = part;
    }

    public int getPosition() {
        return position;
    }

    public void setPosition(int position) {
        this.position = position;
    }
}
