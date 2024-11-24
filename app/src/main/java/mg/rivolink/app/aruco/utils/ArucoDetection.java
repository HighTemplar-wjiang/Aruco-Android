package mg.rivolink.app.aruco.utils;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.Dictionary;
import org.opencv.aruco.DetectorParameters;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.Future;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Callable;

public class ArucoDetection {

    private Mat gray;
    private Dictionary dictionary;
    private DetectorParameters parameters;
    private MatOfInt ids;
    private List<Mat> corners;
    private ThreadPoolExecutor executor;
    private List<Double> alphaValues;
    private List<Mat> brightGrays;
    private List<MatOfInt> tempIdsList;
    private List<List<Mat>> tempCornersList;
    private List<Mat> invertGrays;
    private List<MatOfInt> invTempIdsList;
    private List<List<Mat>> invTempCornersList;

    public ArucoDetection(Dictionary dictionary, DetectorParameters parameters) {
        this.dictionary = dictionary;
        this.parameters = parameters;
        this.ids = new MatOfInt();
        this.corners = new ArrayList<>();
        this.executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(2); // Initialize with 4 threads
        this.alphaValues = new ArrayList<Double>(); // Initialize the list of alpha values
        this.alphaValues.add(0.5);
        this.alphaValues.add(3.0);

        int numThreads = this.executor.getCorePoolSize();
        this.brightGrays = new ArrayList<>(numThreads);
        this.tempIdsList = new ArrayList<>(numThreads);
        this.tempCornersList = new ArrayList<>(numThreads);
        this.invertGrays = new ArrayList<>(numThreads);
        this.invTempIdsList = new ArrayList<>(numThreads);
        this.invTempCornersList = new ArrayList<>(numThreads);

        for (int i = 0; i < numThreads; i++) {
            this.brightGrays.add(new Mat());
            this.tempIdsList.add(new MatOfInt());
            this.tempCornersList.add(new ArrayList<Mat>());
            this.invertGrays.add(new Mat());
            this.invTempIdsList.add(new MatOfInt());
            this.invTempCornersList.add(new ArrayList<Mat>());
        }
    }

    public void detectArucoCodes(Mat grayInput) {
        final Mat gray = grayInput.clone(); // Set the gray image input
        final List<Mat> allCorners = new ArrayList<>();
        final MatOfInt allIds = new MatOfInt();

        List<Future<Void>> futures = new ArrayList<>();

        for (int i = 0; i < alphaValues.size(); i++) {
            final int index = i;
            Future<Void> future = executor.submit(new Callable<Void>() {
                @Override
                public Void call() {
                    // Convert gray image to bright gray image
                    gray.convertTo(brightGrays.get(index), -1, alphaValues.get(index), 0);

                    // Detect markers in bright gray image
                    Aruco.detectMarkers(brightGrays.get(index), dictionary, tempCornersList.get(index), tempIdsList.get(index), parameters);

                    // Calculate the inverted gray image
                    Core.bitwise_not(brightGrays.get(index), invertGrays.get(index));

                    // Detect markers in inverted gray image
                    Aruco.detectMarkers(invertGrays.get(index), dictionary, invTempCornersList.get(index), invTempIdsList.get(index), parameters);

                    synchronized (allCorners) {
                        // Add detected markers to the combined list
                        if (!tempIdsList.get(index).empty() && !tempCornersList.get(index).isEmpty()) {
                            allCorners.addAll(tempCornersList.get(index));
                            allIds.push_back(tempIdsList.get(index));
                        }
                        if (!invTempIdsList.get(index).empty() && !invTempCornersList.get(index).isEmpty()) {
                            allCorners.addAll(invTempCornersList.get(index));
                            allIds.push_back(invTempIdsList.get(index));
                        }
                    }

                    return null;
                }
            });
            futures.add(future);
        }

        for (Future<Void> future : futures) {
            try {
                future.get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }

        // Combine detection results if markers are close
        combineDetections(allIds, allCorners);

        this.ids = allIds;
        this.corners = allCorners;
    }

    public void shutdown() {
        if (executor != null && !executor.isShutdown()) {
            executor.shutdown();
        }
    }

    private void combineDetections(MatOfInt ids, List<Mat> corners) {
        double threshold = 10.0; // Distance threshold to consider markers as close
        List<Integer> combinedIds = new ArrayList<>();
        List<Mat> combinedCorners = new ArrayList<>();

        for (int i = 0; i < ids.rows(); i++) {
            boolean isCombined = false;
            for (int j = 0; j < combinedIds.size(); j++) {
                if (areMarkersClose(corners.get(i), combinedCorners.get(j), threshold)) {
                    isCombined = true;
                    break;
                }
            }
            if (!isCombined) {
                combinedIds.add((int) ids.get(i, 0)[0]); // Cast to int
                combinedCorners.add(corners.get(i));
            }
        }

        // Update ids and corners with combined results
        ids.fromList(combinedIds);
        corners.clear();
        corners.addAll(combinedCorners);
    }

    private boolean areMarkersClose(Mat corners1, Mat corners2, double threshold) {
        double[] center1 = getMarkerCenter(corners1);
        double[] center2 = getMarkerCenter(corners2);
        double distance = Math.sqrt(Math.pow(center1[0] - center2[0], 2) + Math.pow(center1[1] - center2[1], 2));
        return distance < threshold;
    }

    private double[] getMarkerCenter(Mat corners) {
        double[] center = new double[2];
        for (int i = 0; i < corners.rows(); i++) {
            center[0] += corners.get(i, 0)[0];
            center[1] += corners.get(i, 0)[1];
        }
        center[0] /= corners.rows();
        center[1] /= corners.rows();
        return center;
    }

    public MatOfInt getIds() {
        return ids;
    }

    public List<Mat> getCorners() {
        return corners;
    }
}