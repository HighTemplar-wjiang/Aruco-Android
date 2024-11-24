package mg.rivolink.app.aruco;

import android.app.Activity;
import android.content.Intent;

import android.media.MediaRecorder;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Button;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Build;
import android.provider.Settings;
import androidx.annotation.RequiresApi;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

//import android.support.v7.app.AppCompatActivity;

import android.view.WindowManager;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import android.content.ContentValues;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;

import java.io.File;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;

import mg.rivolink.app.aruco.renderer.Renderer3D;
import mg.rivolink.app.aruco.utils.ArucoDetection;
import mg.rivolink.app.aruco.utils.CameraParameters;
import mg.rivolink.app.aruco.view.PortraitCameraLayout;
import mg.rivolink.app.aruco.utils.ArucoParameters;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.DetectorParameters;
import org.opencv.aruco.Dictionary;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import org.rajawali3d.view.SurfaceView;

public class MainActivity extends AppCompatActivity implements CvCameraViewListener2 {

	public static final float SIZE = 0.04f;
	private static final int REQUEST_MANAGE_EXTERNAL_STORAGE = 1;

	private TextView textInfo;
	private Button btnTakePhoto;
	private Button btnRecordVideo;

	// Video recording.
	private MediaRecorder mediaRecorder;
	private boolean isRecording = false;
	private String videoFilePath;

	private Mat cameraMatrix;
	private MatOfDouble distCoeffs;
	private long lastFrameTime;
	private long currentFrameTime;
	private long movingAverageTime;

	private Mat rgb;
	private Mat lastRgb;
	private Mat gray;
	private Mat brightGray;

	private Mat rvecs;
	private Mat tvecs;

	private MatOfInt ids;
	private List<Mat> corners;
	private Dictionary dictionary;
	private Dictionary selectedDictionary;
	private DetectorParameters parameters;
	private static List<Double> alphaList = Arrays.asList(0.2, 0.5, 3.0);

	private Renderer3D renderer;
	private CameraBridgeViewBase camera;
	
	private final BaseLoaderCallback loaderCallback = new BaseLoaderCallback(this){
        @Override
        public void onManagerConnected(int status){
			if(status == LoaderCallbackInterface.SUCCESS){
				Activity activity = MainActivity.this;
				
				cameraMatrix = Mat.eye(3, 3, CvType.CV_64FC1);
				distCoeffs = new MatOfDouble(Mat.zeros(5, 1, CvType.CV_64FC1));
				
				if(CameraParameters.fileExists(activity)){
					CameraParameters.tryLoad(activity, cameraMatrix, distCoeffs);
				}
				else {
					CameraParameters.selectFile(activity);
				}
				
				camera.enableView();
			}
			else {
				super.onManagerConnected(status);
			}
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState){
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.main_layout);

		camera = ((PortraitCameraLayout)findViewById(R.id.camera_layout)).getCamera();
        camera.setVisibility(SurfaceView.VISIBLE);
        camera.setCvCameraViewListener(this);

		renderer = new Renderer3D(this);

		SurfaceView surface = (SurfaceView)findViewById(R.id.main_surface);
		surface.setTransparent(true);
		surface.setSurfaceRenderer(renderer);

		// Initialize TextView for showing the number of ArUco markers detected
		textInfo = findViewById(R.id.text_info);

		// Initialize Take Photo button
		btnTakePhoto = findViewById(R.id.btn_take_photo);
		btnTakePhoto.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				takePhoto();
			}
		});

		// Take video button.
		btnRecordVideo = findViewById(R.id.btn_record_video);
		btnRecordVideo.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				if (isRecording) {
					stopRecording();
					btnRecordVideo.setText("Record Video");
				} else {
					startRecording();
					btnRecordVideo.setText("Stop Recording");
				}
			}
		});

		Spinner dictionarySpinner = findViewById(R.id.dictionarySpinner);

		// Get the available dictionary names from ArucoParameters
		String[] dictionaryNames = ArucoParameters.getAvailableDictionaries();

		// Populate the spinner with dictionary names
		ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, dictionaryNames);
		adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
		dictionarySpinner.setAdapter(adapter);

		// Set a listener for selecting a dictionary
		dictionarySpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
			@Override
			public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
				String selectedDictionaryName = parent.getItemAtPosition(position).toString();

				// Update the selected ArUco dictionary using ArucoParameters
				selectedDictionary = ArucoParameters.getDictionaryByName(selectedDictionaryName);
				dictionary = selectedDictionary;
			}

			@Override
			public void onNothingSelected(AdapterView<?> parent) {
				// Default behavior if nothing is selected
			}
		});

		// Request all files access
		requestAllFileAccess();

		// Set an initial dictionary if needed
//		selectedDictionary = ArucoParameters.getDictionaryByName("DICT_6X6_50");
//		dictionary = selectedDictionary;

	}

	@Override
	protected void onActivityResult(int requestCode, int resultCode, Intent data){
        super.onActivityResult(requestCode, resultCode, data);
        CameraParameters.onActivityResult(this, requestCode, resultCode, data, cameraMatrix, distCoeffs);
		if (requestCode == REQUEST_MANAGE_EXTERNAL_STORAGE) {
			if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
				if (Environment.isExternalStorageManager()) {
					// Permission granted
					Toast.makeText(this, "All files access granted", Toast.LENGTH_SHORT).show();
				} else {
					// Permission denied
					Toast.makeText(this, "All files access denied", Toast.LENGTH_SHORT).show();
				}
			}
		}
	}


	@Override
    public void onResume(){
        super.onResume();

		if(OpenCVLoader.initDebug())
			loaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
		else
			Toast.makeText(this, getString(R.string.error_native_lib), Toast.LENGTH_LONG).show();
    }
	
	@Override
    public void onPause(){
        super.onPause();

        if(camera != null)
            camera.disableView();
    }

	@Override
    public void onDestroy(){
        super.onDestroy();

        if (camera != null)
            camera.disableView();
    }

	@Override
	public void onCameraViewStarted(int width, int height){
		rgb = new Mat();
		lastRgb = new Mat();
		brightGray = new Mat();
		corners = new LinkedList<>();
		parameters = DetectorParameters.create();
		dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_4X4_50);
		lastFrameTime = System.currentTimeMillis();
		// Create an instance of ArucoDetection
//		arucoDetection = new ArucoDetection(dictionary, parameters);
	}

	@Override
	public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame){
		if(!CameraParameters.isLoaded()){
			return inputFrame.rgba();
		}

		String detectCode = "O";
		
		Imgproc.cvtColor(inputFrame.rgba(), rgb, Imgproc.COLOR_RGBA2RGB);
		gray = inputFrame.gray();

		// Perform detection
//		arucoDetection.detectArucoCodes(gray);

		// Retrieve detected IDs and corners
//		ids = arucoDetection.getIds();
//		corners = arucoDetection.getCorners();


		// Create inverse of gray
		Core.bitwise_not(gray, brightGray);

		// Detect markers in gray
		ids = new MatOfInt();
		corners.clear();
		Aruco.detectMarkers(gray, dictionary, corners, ids, parameters);
		if(corners.isEmpty())
		{
			// Detect markers in invGray
			Aruco.detectMarkers(brightGray, dictionary, corners, ids, parameters);
			detectCode = "I";
		}

		// Still empty.
		if(corners.isEmpty())
		{
			for(Double alpha : alphaList)
			{
				gray.convertTo(brightGray, -1, alpha, 0);
				Core.normalize(brightGray, brightGray, 0, 255, Core.NORM_MINMAX);
				Aruco.detectMarkers(brightGray, dictionary, corners, ids, parameters);
				detectCode = String.format(Locale.CHINA,"%.1f", alpha);
				if(!corners.isEmpty()) {
					break;
				}
			}
		}

		long numberOfMarkers = 0;
		if(!corners.isEmpty()){
			Aruco.drawDetectedMarkers(rgb, corners, ids);

			rvecs = new Mat();
			tvecs = new Mat();

			Aruco.estimatePoseSingleMarkers(corners, SIZE, cameraMatrix, distCoeffs, rvecs, tvecs);
			for(int i = 0;i<ids.toArray().length;i++){
				draw3dCube(rgb, cameraMatrix, distCoeffs, rvecs.row(i), tvecs.row(i), new Scalar(255, 0, 0));
				Aruco.drawAxis(rgb, cameraMatrix, distCoeffs, rvecs.row(i), tvecs.row(i), SIZE/2.0f);
			}

			numberOfMarkers = ids.total();

		} else {
			// No markers detected, set the count to zero
		}

		// Record frame time.
		currentFrameTime = System.currentTimeMillis();
		if(movingAverageTime == 0)
		{
			movingAverageTime = currentFrameTime - lastFrameTime;
		}
		else {
			movingAverageTime = (long) (movingAverageTime * 0.8 + (currentFrameTime - lastFrameTime) * 0.2);
		}

		// Update UI with the total time for one frame
		final String statusText = numberOfMarkers +  "|" +  movingAverageTime + "ms" + "|" + detectCode;
		runOnUiThread(new Runnable() {
			@Override
			public void run() {
				textInfo.setText(statusText);
			}
		});
		lastFrameTime = currentFrameTime;

		lastRgb = rgb.clone();

		// Write the frame to the MediaRecorder if recording
		if (isRecording) {
			try {
				Bitmap bitmap = Bitmap.createBitmap(lastRgb.cols(), lastRgb.rows(), Bitmap.Config.ARGB_8888);
				org.opencv.android.Utils.matToBitmap(lastRgb, bitmap);

				android.graphics.Canvas canvas = mediaRecorder.getSurface().lockCanvas(null);
				canvas.drawBitmap(bitmap, 0, 0, null);
				mediaRecorder.getSurface().unlockCanvasAndPost(canvas);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		return rgb;
	}

	private void takePhoto() {
		String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
		String fileName = "IMG_" + timeStamp + ".jpg";

		try {
			// Convert the modified RGB Mat to Bitmap
			Bitmap bitmap = Bitmap.createBitmap(lastRgb.cols(), lastRgb.rows(), Bitmap.Config.ARGB_8888);
			org.opencv.android.Utils.matToBitmap(lastRgb, bitmap);

			// Prepare content values for MediaStore
			ContentValues values = new ContentValues();
			values.put(MediaStore.Images.Media.DISPLAY_NAME, fileName);
			values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
			values.put(MediaStore.Images.Media.DATE_ADDED, System.currentTimeMillis() / 1000);
			values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/MyApp"); // Save to Pictures/MyApp

			// Insert image into MediaStore
			Uri uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);

			if (uri != null) {
				try (OutputStream outStream = getContentResolver().openOutputStream(uri)) {
					if (outStream != null) {
						bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outStream);
						Toast.makeText(this, "Photo saved to Photos: " + uri.toString(), Toast.LENGTH_SHORT).show();
					}
				} catch (Exception e) {
					e.printStackTrace();
					Toast.makeText(this, "Failed to save photo", Toast.LENGTH_SHORT).show();
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
			Toast.makeText(this, "Failed to save photo", Toast.LENGTH_SHORT).show();
		}
	}

	private void startRecording() {
		try {
			// Initialize MediaRecorder
			mediaRecorder = new MediaRecorder();
			mediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
			mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);

			// Create a file to save the video
			String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
			String videoFileName = "VID_" + timeStamp + ".mp4";
			File videoFile = new File(getExternalFilesDir(Environment.DIRECTORY_MOVIES), videoFileName);
			videoFilePath = videoFile.getAbsolutePath();

			mediaRecorder.setOutputFile(videoFilePath);
			mediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
			mediaRecorder.setVideoEncodingBitRate(10000000);
			mediaRecorder.setVideoFrameRate(30);
			mediaRecorder.setVideoSize(lastRgb.cols(), lastRgb.rows());

			mediaRecorder.prepare();
			mediaRecorder.start();
			isRecording = true;
		} catch (Exception e) {
			e.printStackTrace();
			Toast.makeText(this, "Failed to start recording", Toast.LENGTH_SHORT).show();
		}
	}

	private void stopRecording() {
		try {
			mediaRecorder.stop();
			mediaRecorder.release();
			mediaRecorder = null;
			isRecording = false;

			// Save the video to the gallery
			ContentValues values = new ContentValues();
			values.put(MediaStore.Video.Media.DISPLAY_NAME, new File(videoFilePath).getName());
			values.put(MediaStore.Video.Media.MIME_TYPE, "video/mp4");
			values.put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_MOVIES + "/MyApp");
			values.put(MediaStore.Video.Media.DATE_ADDED, System.currentTimeMillis() / 1000);
			values.put(MediaStore.Video.Media.DATA, videoFilePath);

			Uri uri = getContentResolver().insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values);
			if (uri != null) {
				Toast.makeText(this, "Video saved to gallery: " + uri.toString(), Toast.LENGTH_SHORT).show();
			} else {
				Toast.makeText(this, "Failed to save video", Toast.LENGTH_SHORT).show();
			}
		} catch (Exception e) {
			e.printStackTrace();
			Toast.makeText(this, "Failed to stop recording", Toast.LENGTH_SHORT).show();
		}
	}

	@Override
	public void onCameraViewStopped(){
		rgb.release();
		lastRgb.release();
	}
	
	public void draw3dCube(Mat frame, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec, Scalar color){
		double halfSize = SIZE/2.0;

		List<Point3> points = new ArrayList<Point3>();
		points.add(new Point3(-halfSize, -halfSize, 0));
		points.add(new Point3(-halfSize,  halfSize, 0));
		points.add(new Point3( halfSize,  halfSize, 0));
		points.add(new Point3( halfSize, -halfSize, 0));
		points.add(new Point3(-halfSize, -halfSize, SIZE));
		points.add(new Point3(-halfSize,  halfSize, SIZE));
		points.add(new Point3( halfSize,  halfSize, SIZE));
		points.add(new Point3( halfSize, -halfSize, SIZE));

		MatOfPoint3f cubePoints = new MatOfPoint3f();
		cubePoints.fromList(points);

		MatOfPoint2f projectedPoints = new MatOfPoint2f();
		Calib3d.projectPoints(cubePoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

		List<Point> pts = projectedPoints.toList();

	    for(int i=0; i<4; i++){
	        Imgproc.line(frame, pts.get(i), pts.get((i+1)%4), color, 2);
	        Imgproc.line(frame, pts.get(i+4), pts.get(4+(i+1)%4), color, 2);
	        Imgproc.line(frame, pts.get(i), pts.get(i+4), color, 2);
	    }	        
	}
	
	private void transformModel(final Mat tvec, final Mat rvec){
		runOnUiThread(new Runnable(){
			@Override
			public void run(){
				renderer.transform(
					tvec.get(0, 0)[0]*50,
					-tvec.get(0, 0)[1]*50,
					-tvec.get(0, 0)[2]*50,
				
					rvec.get(0, 0)[2], //yaw
					rvec.get(0, 0)[1], //pitch
					rvec.get(0, 0)[0] //roll
				);
			}
		});
	}

	@RequiresApi(api = Build.VERSION_CODES.R)
	private void requestAllFileAccess() {
		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
			if (!Environment.isExternalStorageManager()) {
				Intent intent = new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION);
				startActivityForResult(intent, REQUEST_MANAGE_EXTERNAL_STORAGE);
			}
		} else {
			if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
				ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_MANAGE_EXTERNAL_STORAGE);
			}
		}
	}


	@Override
	public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
		super.onRequestPermissionsResult(requestCode, permissions, grantResults);
		if (requestCode == REQUEST_MANAGE_EXTERNAL_STORAGE) {
			if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
				// Permission granted
			} else {
				// Permission denied
			}
		}
	}
	
}


