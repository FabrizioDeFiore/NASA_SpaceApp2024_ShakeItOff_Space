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
    public bool QuakeOn = false;         // Boolean to control whether quake is currently on
    public bool QuakeStart = false;      // Boolean to control whether quake starts

    void Start()
    {
        // Save the original position of the object
        originalPosition = transform.localPosition;
    }

    public void StartTremor()
    {
        // Begin tremor
        tremorTime = tremorDuration;
        QuakeOn = true;  // Start the quake
    }

    void Update()
    {
        if (QuakeStart)
        {
            StartTremor();
            QuakeStart = false;
        }
        if (QuakeOn && tremorTime > 0)
        {
            // Calculate the current intensity based on the elapsed time
            float normalizedTime = 1 - (tremorTime / tremorDuration);
            float currentIntensity;

            if (normalizedTime < 0.5f)
            {
                // Accelerate to the set intensity
                currentIntensity = tremorIntensity * (normalizedTime * 2);
            }
            else
            {
                // Decelerate to 0
                currentIntensity = tremorIntensity * ((1 - normalizedTime) * 2);
            }

            // Shake the object by randomizing its position slightly
            transform.localPosition = originalPosition + UnityEngine.Random.insideUnitSphere * currentIntensity;

            // Countdown the tremor timer
            tremorTime -= Time.deltaTime;
        }
        else if (tremorTime <= 0 && QuakeOn)
        {
            // Reset the position once the tremor ends
            transform.localPosition = originalPosition;
            QuakeOn = false;  // Stop the quake after it finishes
        }
    }

    public void ResetQuake()
    {
        QuakeOn = false;
        transform.localPosition = originalPosition;
    }
}