Shader "Custom/PulsatingHeatmapWithTransparency"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Center ("Center Position", Vector) = (0.5, 0.5, 0, 0)
        _Radius ("Radius", Float) = 0.5
        _ColorHot ("Hot Color", Color) = (1, 0, 0, 1)  // Red at the center
        _ColorCold ("Cold Color", Color) = (0, 0, 1, 1)  // Blue at the middle
        _PulseDuration ("Pulse Duration", Float) = 1.0  // Duration of a single pulse
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }
        LOD 200

        Pass
        {
            Blend SrcAlpha OneMinusSrcAlpha  // Enable blending for transparency
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _Center;
            float _Radius;
            float _PulseDuration;
            float4 _ColorHot;
            float4 _ColorCold;

            // A function to calculate pulsating effect based on time
            float PulsatingAlpha(float time)
            {
                float pulse = sin(time * (2.0 * 3.14159 / _PulseDuration));
                return saturate((pulse + 1.0) * 0.5); // Range [0, 1]
            }

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // Calculate distance from the center
                float2 uv = i.uv;
                float dist = distance(uv, _Center.xy);

                // Normalize distance for color transition
                float normalizedDist = saturate(dist / _Radius);

                // Interpolate between hot and cold colors based on distance
                float4 color = lerp(_ColorHot, _ColorCold, normalizedDist);

                // Get the pulsating alpha value based on time
                float pulseAlpha = PulsatingAlpha(_Time.y);

                // Calculate final alpha based on distance and pulsating effect
                // The alpha should be 1 (opaque) at the center and 0 (transparent) at the edges
                float alpha = pulseAlpha * saturate(1.0 - normalizedDist);

                // Set the final output color with transparency
                return float4(color.rgb, alpha);
            }
            ENDCG
        }
    }
    FallBack "Transparent"
}
