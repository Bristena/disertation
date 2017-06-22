package ui;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ChooseFile extends JFrame {

    public ChooseFile() throws HeadlessException {
        super("The best app ever!!!");
        setSize(350, 200);
        setDefaultCloseOperation(EXIT_ON_CLOSE);

        final Container c = getContentPane();
        c.setLayout(new FlowLayout());

        JButton openButton = new JButton("Open");
        JButton predictButton = new JButton("Predict");
        final JLabel imageLabel = new JLabel();
        final JLabel statusbar =
                new JLabel("Output of your selection will go here");

        // Create a file chooser that opens up as an Open dialog
        openButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae) {
                JFileChooser chooser = new JFileChooser();
                chooser.setMultiSelectionEnabled(true);
                int option = chooser.showOpenDialog(ChooseFile.this);
                if (option == JFileChooser.APPROVE_OPTION) {
                    File[] sf = chooser.getSelectedFiles();
                    String filePath = sf[0].getAbsolutePath();
                    statusbar.setText("You chose " + filePath);
                }
            }
        });

        predictButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae) {
                BufferedImage wPic = null;
                try {
                    wPic = ImageIO.read(new File(""));
//                   imageLabel.setIcon(wPic);
                    c.add(imageLabel);
                } catch (IOException e) {
                    System.out.println(e);                }
            }
        });

        c.add(openButton);
        c.add(predictButton);
        c.add(statusbar);
    }

    public static void main(String[] args) {
        ChooseFile chooseFile = new ChooseFile();
        chooseFile.setVisible(true);
    }
}
