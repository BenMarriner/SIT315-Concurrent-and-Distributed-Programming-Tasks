// C++ code
//

// Pin numbers
const int MOTION_PIN = 2;
const int BUTTON_PIN = 3;
const int BUZZER_PIN = 4;
const int LED_PIN = 5;

bool motionDetected = false; // Has motion been detected?
bool ledState = false; // State of LED
bool buttonPushed = false; // State of button

void setup()
{
  // Configure pin modes
  pinMode(MOTION_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  
  // Attach interrupts
  attachInterrupt(digitalPinToInterrupt(MOTION_PIN), buzzer_ISR, RISING); // Interrupt for sounding alarm
  attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), toggle_led_ISR, RISING); // Interrupt for button press
  
  // Initialise serial monitor
  Serial.begin(9600);
}

// The loop keeps track of whether motion has been detected and any serial monitor logging messages
// The buzzer will still be triggered by an interrupt caused by the motion sensor
void loop()
{
  motionDetected = digitalRead(MOTION_PIN) == 1;
  buttonPushed = digitalRead(BUTTON_PIN) == 1;
  
  Serial.println(motionDetected ? "Motion detected!" : "No motion detected");
  Serial.println(buttonPushed ? "Button pushed" : "Button released");
  Serial.println(ledState ? "LED on" : "LED off");

  delay(1000);
}

// Interrupt function for sounding alarm
void buzzer_ISR()
{
  tone(BUZZER_PIN, 523, 3000); // play tone 60 (C5 = 523 Hz)
}

// Interrupt function for toggling LED
void toggle_led_ISR()
{
  ledState = !ledState;
  digitalWrite(LED_PIN, ledState);
}
