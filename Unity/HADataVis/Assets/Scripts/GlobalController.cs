using UnityEngine;
using UnityEngine.UI;

[System.Serializable]
public class TremorEvent
{
    public Texture2D picture;         // Picture associated with the tremor event
    public float duration;            // Duration of the tremor
    public float intensity;           // Intensity of the tremor
    public float startPercentage;     // Percentage at which the tremor appears on the graph
    public double magnitude;           // Magnitude of the tremor
    public double velocity;            // Velocity of the tremor    
}

//Select the raw image to change the texture



public class GlobalController : MonoBehaviour
{
    public DataScript dataScript;
    public QuakeScript quakeScript;
    public RawImage rawImage;
    public AnimControl animControl;
    public AnimControl animControl2;
    public TremorEvent[] tremorEvents; // Array of tremor events
    int i = 0;
    int j = 0;
    float progress = 0.0f;
    bool quakeHasStarted = false;
    bool play;
    void Start()
    {
        rawImage.texture = tremorEvents[i].picture;
        dataScript.magnitude = tremorEvents[i].magnitude;
        dataScript.velocity = tremorEvents[i].velocity;
        quakeScript.tremorIntensity = tremorEvents[i].intensity;
        quakeScript.tremorDuration = tremorEvents[i].duration;

    }

    // Update is called once per frame
    void Update()
    {
        progress = animControl.progress;

        if (Input.GetKeyDown(KeyCode.RightArrow))
        {
            j = 0;
            animControl.resetAnimation = true;
            animControl2.resetAnimation = true;
            i++;
            if (i >= tremorEvents.Length)
            {
                i = 0; // Loop back to the first event
            }
            quakeHasStarted = false;
            UpdateTremorEvent();
        }

        if (Input.GetKeyDown(KeyCode.LeftArrow))
        {
            j = 0;
            animControl.resetAnimation = true;
            animControl2.resetAnimation = true;

            if (i == 0)
            {
                i = tremorEvents.Length - 1; // Loop back to the last event
            }
            else
            {
                i--;
            }
            quakeHasStarted = false;
            UpdateTremorEvent();
        }

        if (progress > tremorEvents[i].startPercentage && quakeHasStarted)
        {
            quakeScript.StartTremor();
            quakeHasStarted = false;
        }

        if (Input.GetKeyDown(KeyCode.Space))
        {
            animControl.ResetAnimation();
            animControl2.ResetAnimation();
            if (j == 1)
            {
                animControl.ResetAnimation();
                animControl2.ResetAnimation();
                quakeScript.ResetQuake();
                j = 0;
            }
            else
            {
                animControl.StartAnimation();         
                animControl2.StartAnimation();                 
                quakeHasStarted = true;
                j = 1;
            }
        }
            
    }

    void UpdateTremorEvent()
    {
        rawImage.texture = tremorEvents[i].picture;
        dataScript.magnitude = tremorEvents[i].magnitude;
        dataScript.velocity = tremorEvents[i].velocity;
        quakeScript.tremorIntensity = tremorEvents[i].intensity;
        quakeScript.tremorDuration = tremorEvents[i].duration;
        animControl.ResetAnimation();
        quakeScript.ResetQuake();
    }
}