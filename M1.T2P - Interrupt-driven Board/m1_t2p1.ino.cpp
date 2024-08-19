// C++ code
//

// Pin numbers
const int MOTION_PIN = 2;
const int BUZZER_PIN = 3;

// Has motion been detected?
bool motionDetected = false;

void setup()
{
  // Configure pin modes
  pinMode(MOTION_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  
  // Attach interrupts
  attachInterrupt(digitalPinToInterrupt(MOTION_PIN), buzzer_ISR, RISING); // Interrupt for sounding alarm
  
  // Initialise serial monitor
  Serial.begin(9600);
}

// The loop keeps track of whether motion has been detected and any serial monitor logging messages
// The buzzer will still be triggered by an interrupt caused by the motion sensor
void loop()
{
  motionDetected = digitalRead(MOTION_PIN) == 1;
  
  if (motionDetected)
    Serial.println("Motion detected!");
  else
    Serial.println("No motion detected");
  
  delay(1000);
}

// Interrupt function for sounding alarm
void buzzer_ISR()
{
  tone(BUZZER_PIN, 523, 3000); // play tone 60 (C5 = 523 Hz)
}
