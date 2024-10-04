using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class QuakeScript : MonoBehaviour
{
    // Adjustable intensity and duration
    public float tremorIntensity = 0.1f; // How strong the tremor is
    public float tremorDuration = 1.0f;  // How long the tremor lasts
    private Vector3 originalPosition;    // To store the object's original position

    private float tremorTime = 0f;       // Timer for tremor duration
    public bool startQuake = false;      // Boolean to control whether quake starts

    void Start()
    {
        // Save the original position of the object
        originalPosition = transform.localPosition;
    }

    public void StartTremor()
    {
        // Begin tremor
        tremorTime = tremorDuration;
        startQuake = true;  // Start the quake
    }

    void Update()
    {
        if (startQuake && tremorTime > 0)
        {
            // Shake the object by randomizing its position slightly
            transform.localPosition = originalPosition + UnityEngine.Random.insideUnitSphere * tremorIntensity;

            // Countdown the tremor timer
            tremorTime -= Time.deltaTime;
        }
        else if (tremorTime <= 0 && startQuake)
        {
            // Reset the position once the tremor ends
            transform.localPosition = originalPosition;
            startQuake = false;  // Stop the quake after it finishes
        }
    }
}
