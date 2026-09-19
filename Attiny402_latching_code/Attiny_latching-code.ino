#include <Arduino.h>
#include <avr/sleep.h>

// =====================================================
// ATtiny402 PIN ASSIGNMENT
// =====================================================

#define BUTTON_PIN       PIN_PA1   // Push button -> GND
#define MOSFET_PIN       PIN_PA2   // AO3407A Gate

#define PI_BUTTON_PIN    PIN_PA6   // PA6 -> Raspberry Pi GPIO17
#define PI_SHUTDOWN_PIN  PIN_PA7   // Raspberry Pi GPIO21 -> PA7

#define OFF_DELAY        3000UL    // 3 seconds


// =====================================================
// STATE
// =====================================================

bool piOn = false;

bool shutdownTimer = false;
unsigned long shutdownStart = 0;


// =====================================================
// ATtiny402 PORT INTERRUPT
//
// PA1 falling edge wakes ATtiny from sleep.
// =====================================================

ISR(PORTA_PORT_vect)
{
    // Clear PA1 interrupt flag
    PORTA.INTFLAGS = PIN1_bm;
}


// =====================================================
// SEND LIVE BUTTON SIGNAL TO RASPBERRY PI GPIO17
//
// BUTTON PRESSED:
//     PA6 pulls GPIO17 LOW
//
// BUTTON RELEASED:
//     PA6 becomes HIGH-Z
//     Pi 100k pull-up makes GPIO17 HIGH
// =====================================================

void sendButtonSignal(bool pressed)
{
    if (pressed)
    {
        // Pull Raspberry Pi GPIO17 LOW
        pinMode(PI_BUTTON_PIN, OUTPUT);
        digitalWrite(PI_BUTTON_PIN, LOW);
    }
    else
    {
        // Release GPIO17
        // Raspberry Pi 100k pull-up makes it HIGH
        pinMode(PI_BUTTON_PIN, INPUT);
    }
}


// =====================================================
// ENTER POWER-DOWN SLEEP
//
// MOSFET is already OFF.
// PA1 button wakes the ATtiny.
// =====================================================

void enterSleep()
{
    // Make absolutely sure MOSFET is OFF
    digitalWrite(MOSFET_PIN, HIGH);

    // Release GPIO17
    pinMode(PI_BUTTON_PIN, INPUT);

    // PA7 = shutdown input
    pinMode(PI_SHUTDOWN_PIN, INPUT);

    // PA1 = button input with pull-up
    pinMode(BUTTON_PIN, INPUT_PULLUP);


    // -------------------------------------------------
    // Configure PA1 interrupt
    //
    // LOW when button pressed
    // Falling edge wakes ATtiny
    // -------------------------------------------------

    PORTA.PIN1CTRL =
        PORT_PULLUPEN_bm |
        PORT_ISC_FALLING_gc;

    // Clear any old PA1 interrupt flag
    PORTA.INTFLAGS = PIN1_bm;


    // -------------------------------------------------
    // Configure Power-Down sleep
    // -------------------------------------------------

    set_sleep_mode(SLEEP_MODE_PWR_DOWN);

    sleep_enable();

    // Enable global interrupts
    sei();

    // -------------------------------------------------
    // SLEEP
    // -------------------------------------------------

    sleep_cpu();

    // -------------------------------------------------
    // ATtiny wakes here after PA1 button press
    // -------------------------------------------------

    sleep_disable();

    delay(5);
}


// =====================================================
// SETUP
// =====================================================

void setup()
{
    // -------------------------------------------------
    // AO3407A
    //
    // PA2 LOW  = MOSFET ON
    // PA2 HIGH = MOSFET OFF
    // -------------------------------------------------

    digitalWrite(MOSFET_PIN, HIGH);
    pinMode(MOSFET_PIN, OUTPUT);


    // -------------------------------------------------
    // Physical push button
    //
    // PA1 -> BUTTON -> GND
    // -------------------------------------------------

    pinMode(BUTTON_PIN, INPUT_PULLUP);


    // -------------------------------------------------
    // PA6 -> Raspberry Pi GPIO17
    //
    // GPIO17 has 100k pull-up to 3.3V
    // -------------------------------------------------

    pinMode(PI_BUTTON_PIN, INPUT);


    // -------------------------------------------------
    // Raspberry Pi GPIO21 -> PA7
    //
    // PA7 ONLY READS GPIO21
    // -------------------------------------------------

    pinMode(PI_SHUTDOWN_PIN, INPUT);


    // -------------------------------------------------
    // PA1 interrupt configuration
    // -------------------------------------------------

    PORTA.PIN1CTRL =
        PORT_PULLUPEN_bm |
        PORT_ISC_FALLING_gc;

    PORTA.INTFLAGS = PIN1_bm;


    delay(20);
}


// =====================================================
// MAIN LOOP
// =====================================================

void loop()
{
    // =================================================
    // PI / POWER IS OFF
    // =================================================

    if (!piOn)
    {
        // ---------------------------------------------
        // Wait for physical button
        // ---------------------------------------------

        if (digitalRead(BUTTON_PIN) == LOW)
        {
            // =========================================
            // SINGLE PRESS -> POWER ON
            // =========================================

            digitalWrite(MOSFET_PIN, LOW);

            piOn = true;

            shutdownTimer = false;


            // -----------------------------------------
            // Wait for button release
            //
            // This guarantees one press = one ON.
            // -----------------------------------------

            while (digitalRead(BUTTON_PIN) == LOW)
            {
                delay(5);
            }

            delay(100);

            // -----------------------------------------
            // DO NOT SLEEP
            //
            // Pi is now ON.
            // -----------------------------------------

            return;
        }


        // ---------------------------------------------
        // No button -> sleep
        // ---------------------------------------------

        enterSleep();

        return;
    }


    // =================================================
    // PI IS ON
    // =================================================

    bool buttonPressed =
        (digitalRead(BUTTON_PIN) == LOW);


    // =================================================
    // SEND LIVE BUTTON STATE TO GPIO17
    // =================================================

    sendButtonSignal(buttonPressed);


    // =================================================
    // READ PI GPIO21 THROUGH PA7
    //
    // GPIO21 LOW  = normal operation
    // GPIO21 HIGH = shutdown command
    // =================================================

    bool shutdownCommand =
        (digitalRead(PI_SHUTDOWN_PIN) == HIGH);


    // =================================================
    // SHUTDOWN COMMAND
    // =================================================

    if (shutdownCommand)
    {
        // ---------------------------------------------
        // Start 3-second timer
        // ---------------------------------------------

        if (!shutdownTimer)
        {
            shutdownTimer = true;
            shutdownStart = millis();
        }


        // ---------------------------------------------
        // GPIO21 HIGH continuously for 3 seconds
        // ---------------------------------------------

        if (millis() - shutdownStart >= OFF_DELAY)
        {
            // =========================================
            // TURN MOSFET OFF
            // =========================================

            digitalWrite(MOSFET_PIN, HIGH);

            // Give MOSFET gate time to settle
            delay(20);


            // =========================================
            // Update state
            // =========================================

            piOn = false;
            shutdownTimer = false;


            // Release GPIO17
            pinMode(PI_BUTTON_PIN, INPUT);


            // =========================================
            // ENTER POWER-DOWN
            // =========================================

            enterSleep();

            return;
        }
    }
    else
    {
        // GPIO21 LOW
        // Cancel shutdown timer

        shutdownTimer = false;
    }


    delay(5);
}