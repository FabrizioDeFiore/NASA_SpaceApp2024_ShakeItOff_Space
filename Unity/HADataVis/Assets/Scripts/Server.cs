using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text.RegularExpressions;
using System.Text;
using System.Threading;
using System.Collections.Generic;
using TMPro; 
public class MyListenerRealTimeComunicationsV4 : MonoBehaviour
{ 
    public int port = 25002;

    // If object to be update in position or anything else, init here 
    public Transform objectToUpdate1; 
    public Transform objectToUpdate2;
    public Transform objectToUpdate3;

    // Text obj to be updated
    public TMP_Text filenameText;
    public TMP_Text timeRelText;
    public TMP_Text velText;

    public string filename;
    public string timeRel;
    public string vel;

    private Thread receiveThread;
    private TcpListener server;
    private TcpClient client;
    private bool isRunning = true;

    //private float a;
    // List to store all ypos and yscale data points
    // public List<float> yposValues = new List<float>();
    // public List<float> yscaleValues = new List<float>();

    // private List<float> coefficients;


    // Queues for each data stream
    private Queue<float> filenameQueue = new Queue<float>();
    private Queue<float> timeRelQueue = new Queue<float>();
    private Queue<float> velQueue = new Queue<float>();

    void Start(){
        if (objectToUpdate1 == null || objectToUpdate2 == null || objectToUpdate3 == null ){
            Debug.LogError("Assign all three objects in the Inspector!");
            return;
        }
        receiveThread = new Thread(ReceiveData);
        receiveThread.IsBackground = true;
        receiveThread.Start();
        // Calculate the polynomial coefficients
        //coefficients = CalculateCoefficients();

    }

    private void ReceiveData(){
        server = new TcpListener(IPAddress.Any, port);
        server.Start();
        while (isRunning){
            try{
                client = server.AcceptTcpClient();
                using (NetworkStream stream = client.GetStream()){
                    byte[] buffer = new byte[client.ReceiveBufferSize];
                    StringBuilder dataBuilder = new StringBuilder();
                    while (isRunning){
                        int bytesRead = stream.Read(buffer, 0, client.ReceiveBufferSize);
                        if (bytesRead == 0){
                            // Client disconnected
                            Debug.Log("Breaking connection to client");
                            break;
                        }
                        string dataString = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                        dataBuilder.Append(dataString);
                        // Process data when a newline is received
                        if (dataBuilder.ToString().EndsWith("\n")){
                            string[] dataLines = dataBuilder.ToString().Split('\n');
                            foreach (string line in dataLines){
                                if (line.Trim() != ""){ // Ignore empty lines
                                    string[] dataParts = Regex.Split(line.Trim(), @"\s+"); // Split by whitespace
                                    if (dataParts.Length >= 2){
                                        // Expect two parts: data stream name and data point
                                        string dataStreamName = dataParts[0];
                                        float dataPoint;
                                        // Attempt to parse data point as float
                                        if (float.TryParse(dataParts[1], out dataPoint)){
                                            // Enqueue data point based on the data stream name
                                            if (dataStreamName == "Filename"){
                                                filenameQueue.Enqueue(dataPoint);
                                            }else if (dataStreamName == "TimeRel"){
                                                timeRelQueue.Enqueue(dataPoint);
                                            }else if (dataStreamName == "Velocity"){
                                                velQueue.Enqueue(dataPoint);
                                            }else{
                                                Debug.LogWarning("Unknown data stream name: " + dataStreamName);
                                            }
                                        }else{
                                            Debug.LogError("Failed to parse data point: " + line);
                                        }
                                    }else{
                                        Debug.LogError("Invalid data format: " + line);
                                    }
                                }
                            }
                            dataBuilder.Clear(); // Reset for the next frame
                        }
                    }
                }
            }catch (Exception e){
                Debug.LogError("Error receiving data: " + e.Message);
            }finally{
                if (client != null){
                    Debug.Log("Finally");
                    client.Close();
                }
            }
        }
        server.Stop();
    }


    //public static (string, string, string) OutputStrings()
    //{
    //    string str1 = filename;
    //    string str2 = timeRel;
    //    string str3 = vel;

    //    // Return the strings as a tuple
    //    return (str1, str2, str3);
    //}



    public static float CalculateYPosition(float yScale){
        // This function calculates the y position using linear interpolation.
        // Ensure yScale is within the valid range (0.0000 to 1.0000)
        yScale = Mathf.Clamp01(yScale);
        // Define minimum and maximum y positions
        float yPosMin = 1.5f;
        float yPosMax = -0.6251f;
        // Calculate the interpolation factor based on yScale
        float interpolationFactor = yScale;
        // Perform linear interpolation to get the final y position
        float yPosition = Mathf.Lerp(yPosMax, yPosMin, interpolationFactor);
        return yPosition;
    }

    public void UpdateObject(Transform objectToUpdate, float yScale, float yPosition){
        // Create a temporary Vector3 with only the updated Y values
        Vector3 updatedPosition = new Vector3(objectToUpdate.localPosition.x, yPosition, objectToUpdate.localPosition.z);
        // Update the object's position with the temporary Vector3
        objectToUpdate.localPosition = updatedPosition;
        // Update the Y scale of the object
        objectToUpdate.localScale = new Vector3(objectToUpdate.localScale.x, yScale, objectToUpdate.localScale.z);
    }

    // This function i s called every frame 
    void Update(){
        // Update object Y positions based on their respective queues
        if (filenameQueue.Count > 0){
            float data = filenameQueue.Dequeue();
            // Normalize data to a value between 0 and 1
            //float normalizedData_yScale = data / 100f;
            //float yPosition = CalculateYPosition(normalizedData_yScale);
            //UpdateObject(objectToUpdate1, normalizedData_yScale, yPosition);
            filename = data.ToString("F2");
            //chpTemp1Text.text = normalizedData_yScale.ToString("F4");
        }
        
        if (timeRelQueue.Count > 0){
            float data = timeRelQueue.Dequeue();
            //float normalizedData_yScale = data / 100f;
            //float yPosition = CalculateYPosition(normalizedData_yScale);
            //UpdateObject(objectToUpdate2, normalizedData_yScale, yPosition);
            timeRel = data.ToString("F2")+ (" s");
            //loopTemp1Text.text = normalizedData_yScale.ToString("F4");
        }

        if (velQueue.Count > 0){
            float data = velQueue.Dequeue();
            //float normalizedData_yScale = data * 2.5f  ;  // Just because it's a very smalll value, so this make it the filling nrmalized to the others, change this value as needed
            //float yPosition = CalculateYPosition(normalizedData_yScale);
            //UpdateObject(objectToUpdate3, normalizedData_yScale, yPosition);
            vel = (data).ToString("F2")+ (" m/s"); 
            //stoomText.text = normalizedData_yScale.ToString("F4");
        }
    }
}



            // Normalize data to a value between 0 and 1
            // float normalizedData = data / 100f;
            // // Calculate y scale based on formula
            // float yScale = Mathf.Clamp01(normalizedData * 0.99f); // Adjust 0.99 for a slight gap at 100
            // // Calculate y position based on formula (replace with your specific data points)
            // float yPosition = 0.01f + (normalizedData * (1.06f - (-2.1f))); // Adjust formula based on your data points
            // // Update cylinder's transform
            // objectToUpdate1.transform.localScale = new Vector3(1, yScale, 1);
            // objectToUpdate1.transform.position = new Vector3(0, yPosition, 0);
            // Ensure yscale is within the valid range (0 to 1)
            



    // public void UpdateYpos(float yscale, Transform objectToUpdate){
    //     // Ensure yscale is within the valid range (0 to 1)
    //     yscale = Mathf.Clamp01(yscale);
    //     // Calculate ypos using the polynomial formula with pre-calculated coefficients
    //     float ypos = 0;
    //     for (int i = 0; i < coefficients.Count; i++){
    //         ypos += coefficients[i] * Mathf.Pow(yscale, i);
    //     }
    //     // Update the cylinder's local y position
    //     objectToUpdate.localScale = new Vector3(objectToUpdate.localScale.x, ypos, objectToUpdate.localScale.z);
    // }

    // private List<float> CalculateCoefficients(){
    //     int numPoints = yposValues.Count;
    //     float[,] matrix = new float[numPoints, numPoints + 1];
    //     // Fill the matrix with data points
    //     for (int i = 0; i < numPoints; i++){
    //         for (int j = 0; j < numPoints + 1; j++){
    //             if (j == 0){
    //                 matrix[i, j] = 1;
    //             }else{
    //                 matrix[i, j] = Mathf.Pow(yscaleValues[i], j - 1);
    //             }
    //         }
    //     }
    //     // Solve the system of equations for coefficients using Gaussian elimination
    //     List<float> coefficients = SolveLinearSystem(matrix);
    //     return coefficients;
    // }

    // private List<float> SolveLinearSystem(float[,] matrix){
    //     int numPoints = matrix.GetLength(0);
    //     // Perform Gaussian elimination
    //     for (int col = 0; col < numPoints - 1; col++){
    //         for (int row = col + 1; row < numPoints; row++){
    //             float factor = matrix[row, col] / matrix[col, col];
    //             for (int i = col; i < numPoints + 1; i++){
    //                 matrix[row, i] -= matrix[col, i] * factor;
    //             }
    //         }
    //     }
    //     // Back substitution to solve for coefficients
    //     List<float> coefficients = new List<float>();
    //     for (int i = numPoints - 1; i >= 0; i--){
    //         float sum = 0;
    //         for (int j = i + 1; j < numPoints; j++){
    //             sum += coefficients[j] * matrix[i, j];
    //         }
    //         coefficients.Insert(0, (matrix[i, numPoints] - sum) / matrix[i, i]);
    //     }
    //     return coefficients;
    // }

    // void UpdateObjectTransform(float yscale, GameObject objectToUpdate){
    //     yscale = Mathf.Clamp01(yscale);
    //         // Calculate ypos based on the formula
    //         float ypos = a * yscale * yscale * yscale + b * yscale * yscale + c * yscale + d;
    //         // Update the cylinder's local y position
    //         objectToUpdate.localScale = new Vector3(objectToUpdate.localScale.x, ypos, objectToUpdate.localScale.z);
    //}