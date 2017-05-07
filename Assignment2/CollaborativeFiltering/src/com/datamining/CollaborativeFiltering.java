package com.datamining;

import java.io.*;
import java.util.*;

/**
 * Created by nithinm on 5/3/2017.
 */
public class CollaborativeFiltering {

    private static Map<Integer, Map<Integer, Double>> userData = new HashMap<>();
    private static Map<Integer, Map<Integer, Double>> movieData = new HashMap<>();
    private static Map<Integer, Double> userAvg = new HashMap<>();
    private static Double rmse = 0.0;
    private static Double mae = 0.0;
    private static Double maesum = 0.0;
    private static Double rmsesum = 0.0;
    private static Integer testDataCount = 0;
    private static Integer userIter = 0;
    private static Map<Integer, Integer> usermap = new HashMap<>();
    private static Double[][] weights = new Double[28978][28978];
    private static PrintWriter writer;
    private static String root = "E:\\CollaborativeFiltering"; // change this when changing src code root

    public static void main(String[] args) {
        root = System.getProperty("user.dir");
        loadTrainingData(root + "\\src\\com\\datamining\\TrainingRatings.txt");
        doPostProcessing();
        System.out.println("Training done");
        evaluateOnTestData(root + "\\src\\com\\datamining\\TestingRatings.txt");
        printStats();
    }

    // Training Data Load Section
    // Loads training data from a file and adds it to the in-mem data set
    private static void loadTrainingData(String csvFile) {
        String line = "";
        String delimiter = ",";

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {

            while ((line = br.readLine()) != null) {

                // use comma as separator
                String[] data = line.split(delimiter);
                addDataPoint(Integer.parseInt(data[1]), Integer.parseInt(data[0]), Double.parseDouble(data[2]));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // add a training data point to the in-mem data set
    private static void addDataPoint(Integer userId, Integer movieId, Double rating) {
        if (!userData.containsKey(userId)) {
            userData.put(userId, new HashMap<Integer, Double>());
        }

        if(userData.get(userId).containsKey(movieId)) {
            // this is a dup insert. ignore
        } else {
            userData.get(userId).put(movieId, rating);
        }

        if (!movieData.containsKey(movieId)) {
            movieData.put(movieId, new HashMap<Integer, Double>());
        }

        if(movieData.get(movieId).containsKey(userId)) {
            // this is a dup insert. ignore
        } else {
            movieData.get(movieId).put(userId, rating);
        }
    }
    // Training Data Load Section - End

    // Traning data post processing section
    // Do all post processing of training data
    // Includes computing means and weights
    private static void doPostProcessing() {
        for (Map.Entry<Integer, Map<Integer, Double>> entry : userData.entrySet()) {
            int sum = 0;
            for (Map.Entry<Integer, Double> mv : entry.getValue().entrySet()) {
                sum += mv.getValue();
            }
            userAvg.put(entry.getKey(), 1.0 * sum / entry.getValue().size());
            if (!usermap.containsKey(entry.getKey())) {
                usermap.put(entry.getKey(), userIter++);
            }
        }

        System.out.println("Means calculated.");

        String filePathString = root + "\\src\\com\\datamining\\weights.txt";
        File f = new File(filePathString);

        if(!f.exists()) {
            int count = 0;
            try {
                writer = new PrintWriter(filePathString, "UTF-8");
            } catch (Exception e) {

            }

            Integer u1 = 0;
            Integer u2 = 0;
            Double u1mean, u2mean, w;
            for (Map.Entry<Integer, Map<Integer, Double>> entry1 : userData.entrySet()) {
                System.out.println("Weights calculated for " + count++ + " entries.");
                u1 = entry1.getKey();
                u1mean = userAvg.get(entry1.getKey());
                for (Map.Entry<Integer, Map<Integer, Double>> entry2 : userData.entrySet()) {
                    u2 = entry2.getKey();
                    u2mean = userAvg.get(u2);
                    if (u1 < u2) {
                        w = calcDistance(u1, u2, u1mean, u2mean);
                        writer.println(u1 + "," + u2 + "," + w);
                        addWeight(u1, u2, w);
                    }
                }
                writer.flush();
                if (count % 100 == 0) {
                    System.gc();
                }
            }

            writer.flush();
        } else {
            loadWeights(filePathString);
        }
    }

    private static Double dist, d1, d2, num;
    // Calculates weight for a given user pair
    private  static Double calcDistance(Integer user1, Integer user2, Double u1mean, Double u2mean) {
        dist = 1e-12;;
        if (user1 == user2) {
            return dist;
        }

        num = 0.0;
        d1 = 1e-12;
        d2 = 1e-12;

        Map<Integer, Double> user2Set = userData.get(user2);
        for (Map.Entry<Integer, Double> entry : userData.get(user1).entrySet()) {
            if (user2Set.containsKey(entry.getKey())) {
                num += (entry.getValue() - u1mean) *
                        (user2Set.get(entry.getKey()) - u2mean);
                d1 += Math.pow(entry.getValue() - u1mean, 2);
                d2 += Math.pow(user2Set.get(entry.getKey()) - u2mean, 2);
            }
        }

        dist = num / Math.sqrt(d1 * d2);

        if (dist > 1 || dist < -1) {
            System.out.println("Value out of range:" + dist + "," + user1 + "," + user2);
        }
        return dist;
    }

    // add an user pair weight
    private static void addWeight(int user1, int user2, double weight) {
        weights[usermap.get(user1)][usermap.get(user2)] = weight;
    }

    // load weights from a file
    private static void loadWeights(String csvFile) {
        String line = "";
        String delimiter = ",";

        int count = 0;

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String[] data;
            while ((line = br.readLine()) != null) {
                if (count % 1000000 == 0) {
                    System.out.println("Weights calculated for " + count + " entries.");
                }

                count++;
                // use comma as separator
                data = line.split(delimiter);
                addWeight(Integer.parseInt(data[0]), Integer.parseInt(data[1]), Double.parseDouble(data[2]));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    // End of pros processing section

    // Start of Evaluate Test Data section
    // Do evaluation on test data
    private static void evaluateOnTestData(String fileName) {
        String line = "";
        String delimiter = ",";

        int count = 0;

        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {

            while ((line = br.readLine()) != null) {
                count++;

                // use comma as separator
                String[] data = line.split(delimiter);

                final Integer v1, v2;
                v1 = Integer.parseInt(data[1]);
                v2 = Integer.parseInt(data[0]);
                final Double v3 = Double.parseDouble(data[2]);
                addTestDataPoint(v1, v2, v3);

                if (count % 1000 == 0) {
                    System.out.println("Predictions for " + count + " entries.");
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        mae = maesum / testDataCount;
        rmse = Math.sqrt(rmsesum / testDataCount);
    }

    // Predict  on a single test data and get errors out of it
    private static void addTestDataPoint(int userId, int movieId, double actualRating) {
        Double predictedRating = predictVote(userId, movieId);
//        System.out.println("Pred: " + predictedRating + " Actual: " + actualRating);
        testDataCount++;
        maesum += Math.abs(predictedRating - actualRating);
        rmsesum += (predictedRating - actualRating) * (predictedRating - actualRating);
    }

    // Predict vote for a single test data
    private static Double predictVote(Integer userId, Integer movieId) {
        Double expectedRating = 0.0;

        Double distSum = 0.0;

        for (Map.Entry<Integer, Double> entry : movieData.get(movieId).entrySet()) {
            Double dist = getDistance(userId, entry.getKey());
            Double diff = (entry.getValue() - userAvg.get(entry.getKey()));

            expectedRating += dist * diff;
            distSum += Math.abs(dist);
        }

        if (distSum != 0) {
            expectedRating /= distSum;
        }

        if (userAvg.containsKey(userId)) {
            expectedRating += userAvg.get(userId);
        }

        return expectedRating;
    }

    // Get weight for an user pair
    private static Double getDistance(Integer user1, Integer user2) {
        if (user1 < user2) {
            return weights[usermap.get(user1)][usermap.get(user2)];
        } else if (user1 > user2) {
            return weights[usermap.get(user2)][usermap.get(user1)];
        } else {
            return 1e-12;
        }
    }
    // End of Evaluate Test Data section

    // Prints final computed stats
    private static void printStats() {
        System.out.println("Test data count: " + testDataCount);
        System.out.println("Mean Absolute Error: " + mae);
        System.out.println("Root Mean Squared Error: " + rmse);
    }
}
