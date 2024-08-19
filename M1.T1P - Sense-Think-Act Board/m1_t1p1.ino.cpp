// C++ code
//

void setup()
{
  pinMode(A5, INPUT);
  Serial.begin(9600);
  pinMode(2, OUTPUT);
}

void loop()
{
  Serial.println(analogRead(A5));
  if (analogRead(A5) != 0) {
    tone(2, 523, 1000); // play tone 60 (C5 = 523 Hz)
    Serial.println("Motion detected!");
  } else {
    Serial.println("No motion detected");
    noTone(2);
  }
  delay(10); // Delay a little bit to improve simulation performance
}