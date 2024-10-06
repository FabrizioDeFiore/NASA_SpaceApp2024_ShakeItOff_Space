using UnityEngine;
using System;
public class RotateScript : MonoBehaviour
{
    public int maxRotationAngle = 30; // Maximum rotation angle from the original position
    private Quaternion originalRotation;
    private int i;
    System.Random random1;
    System.Random random2;
    System.Random random3;

    void Start()
    {
        random1 = new System.Random();
        // Store the original rotation
        originalRotation = transform.rotation;
        i = 0;
    }

    public void Rotate()
    {
        

        random3 = new System.Random();
        float randomAngle1 = random1.Next(-maxRotationAngle, maxRotationAngle + 1);
        float randomAngle2 = random2.Next(-maxRotationAngle, maxRotationAngle + 1);
        float randomAngle3 = random3.Next(-maxRotationAngle, maxRotationAngle + 1);



        Debug.Log($"Random angle1: {randomAngle1}");
        Debug.Log($"Random angle2: {randomAngle2}");
        Debug.Log($"Random angle3: {randomAngle3}");

        // Calculate the new rotation
        Quaternion targetRotation = Quaternion.Euler(randomAngle1, randomAngle2, randomAngle3) * originalRotation;

        // Restrict the rotation to the maximum angle
        float currentAngle = Quaternion.Angle(originalRotation, targetRotation);
        if (currentAngle > maxRotationAngle)
        {
            targetRotation = Quaternion.RotateTowards(originalRotation, targetRotation, maxRotationAngle);
        }

        // Apply the rotation
        transform.rotation = targetRotation;
    }
    void Update()
    {
        if (i < 15)
        {
            i++;
        }
        
        if (i == 15)
        {
            random2 = new System.Random();
        }
        
    }
}