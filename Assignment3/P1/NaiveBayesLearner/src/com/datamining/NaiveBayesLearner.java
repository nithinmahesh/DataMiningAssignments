package com.datamining;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by nithinm on 5/15/2017.
 */
public class NaiveBayesLearner {
    private static Double[] PriorProbability = new Double[2];
    private static Map<String, Double[]> WordProbabilty = new HashMap<>();
    private static Map<String, Integer[]> WordCount = new HashMap<>();
    private static Integer[] PriorCount = new Integer[] {0, 0};
    private static Integer[] TotalWordCount = new Integer[] {0, 0};
    private static Double smoothingFactor = 1.0;
    private static String root = "E:\\NaiveBayesLearner"; // change this when changing src code root

    public static void main(String[] args) {
        root = System.getProperty("user.dir");
        loadTrainingData(root + "\\src\\com\\datamining\\train");
        calcProbs();
        // evaluateOnTestData("E:\\NaiveBayesLearner\\src\\com\\datamining\\train");
        evaluateOnTestData(root + "\\src\\com\\datamining\\test");
    }

    private static void evaluateOnTestData(String filename) {
        String line = "";
        String delimiter = " ";
        Integer spamCorrect = 0;
        Integer spamWrong = 0;
        Integer hamCorrect = 0;
        Integer hamWrong = 0;

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {

            String res;
            while ((line = br.readLine()) != null) {

                String[] data = line.split(delimiter);
                if(data.length % 2 != 0 || data.length < 4) {
                    System.out.println("Bad data");
                    break;
                }

                res = data[1];

                Integer prediction = predict(data);

                if(res.equalsIgnoreCase("spam")) {
                    if (prediction == 1) {
                        spamCorrect++;
                    } else {
                        spamWrong++;
                    }
                } else if(res.equalsIgnoreCase("ham")) {
                    if (prediction == 0) {
                        hamCorrect++;
                    } else {
                        hamWrong++;
                    }
                } else {
                    System.out.println("Bad data");
                    break;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println(spamCorrect);
        System.out.println(spamWrong);
        System.out.println(hamCorrect);
        System.out.println(hamWrong);

        Double accuracy = 1.0 * (spamCorrect + hamCorrect) /(spamCorrect + spamWrong + hamCorrect + hamWrong);
        Double precision = 1.0 * spamCorrect / (spamCorrect + spamWrong);
        Double recall = 1.0 * spamCorrect / (spamCorrect + hamWrong);
        System.out.println("Accuracy: " +  accuracy);
        System.out.println("precision: " +  precision);
        System.out.println("recall: " +  recall);
    }

    private static Integer predict(String[] data) {
        Double[] finalProb = new Double[]{0.0,0.0};

        for (int i = 0; i < 2; i++) {
            finalProb[i] = Math.log(PriorProbability[i]);

            finalProb[i] += Math.log(getProb(data[0], i));

            for (int j=2; j<data.length; j += 2) {
                finalProb[i] += Integer.parseInt(data[j+1]) * Math.log(getProb(data[j], i));
            }
        }

        return (finalProb[0] > finalProb[1]) ? 0 : 1;
    }

    private static Double getProb(String word, int spam) {
        if (WordProbabilty.containsKey(word)) {
            return WordProbabilty.get(word)[spam];
        } else {
            return 1.0 * smoothingFactor / (WordCount.size() + TotalWordCount[spam]);
        }
    }

    private static void calcProbs() {
        Integer total = 0;
        for (int i = 0; i < 2; i++) {
            total += PriorCount[i];
        }
        for (int i = 0; i < 2; i++) {
            PriorProbability[i] = PriorCount[i] * 1.0 / total;
        }

        Double[] temp;
        for (Map.Entry<String, Integer[]> word: WordCount.entrySet()) {
            temp = new Double[2];

            for (int i = 0; i < 2; i++) {
                temp[i] = (word.getValue()[i] + smoothingFactor) * 1.0 / (WordCount.size() + TotalWordCount[i]);
            }

            WordProbabilty.put(word.getKey(), temp);
//            System.out.println(temp[0] +" "+ temp[1]);
        }

//        System.out.println(PriorProbability[0]);
//        System.out.println(PriorProbability[1]);
    }

    private static void loadTrainingData(String csvFile) {
        String line = "";
        String delimiter = " ";

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {

            String email, res, key;
            Integer value, count, spam;
            Integer[] counts = new Integer[2];
            while ((line = br.readLine()) != null) {

                String[] data = line.split(delimiter);

                count = 0;
                email = data[count++];

                res = data[count++];

                spam = res.equalsIgnoreCase("spam") ? 1 : 0;
                PriorCount[spam]++;

                if (!WordCount.containsKey(email)) {
                    WordCount.put(email, new Integer[]{0, 0});
                }

                for (int i = 0; i < 2; i++) {
                    counts[i] = WordCount.get(email)[i];
                }

                counts[spam]++;

                WordCount.remove(email);
                WordCount.put(email, new Integer[]{counts[0], counts[1]});

                while (data.length > count) {
                    key = data[count++];
                    value = Integer.parseInt(data[count++]);

                    if (!WordCount.containsKey(key)) {
                        WordCount.put(key, new Integer[]{0, 0});
                    }

                    for (int i = 0; i < 2; i++) {
                        counts[i] = WordCount.get(key)[i];
                    }

                    counts[spam] += value;

                    WordCount.remove(key);
                    WordCount.put(key, new Integer[]{counts[0], counts[1]});

                    TotalWordCount[spam] += value;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

//        System.out.println(TotalWordCount[0]);
//        System.out.println(TotalWordCount[1]);
    }

}
