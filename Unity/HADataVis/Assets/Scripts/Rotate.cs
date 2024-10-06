using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Rotate : MonoBehaviour
{
    public RotateScript rotateObject;
    public bool y;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {

            if (y)
            {
                rotateObject.Rotate(); // Rotate the object by 30 degrees
                y = false;
            }

    }
}
