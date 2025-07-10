#include <nrf52.h>
#include <nrf_power.h>  // 可能需要这个头文件
#include <bluefruit.h>
#include <Wire.h>
#include <bluefruit.h>
#include <Adafruit_LittleFS.h>
#include <InternalFileSystem.h>
#include "Adafruit_MLX90393.h"
// #include <nrfx_power.h>
#define num 8
// #define SDA_PIN 14 // 替换为实际连接的引脚
// #define SCL_PIN 12 // 替换为实际连接的引脚
// #define SDA_PIN 14 // 替换为实际连接的引脚
// #define SCL_PIN 12 // 替换为实际连接的引脚
// TwoWire myWire(NRF_TWIM1, nullptr, SPIM1_SPIS1_TWIM1_TWIS1_SPI1_TWI1_IRQn, 14, 12); // 示例引脚为 SDA: 26, SCL: 27
#define WIRE Wire
// #define SDA_PIN 14 // 替换为实际连接的引脚
// #define SCL_PIN 12 // 替换为实际连接的引脚
#define SDA_PIN 25 // 替换为实际连接的引脚
#define SCL_PIN 26 // 替换为实际连接的引脚
// 定义自定义 SDA 和 SCL 引脚
// #define CUSTOM_SDA 28
// #define CUSTOM_SCL 26
// 创建一个自定义的 TwoWire 对象，指定 I2C 硬件实例
Adafruit_MLX90393 sensor[num];
//int CS[num] = {26, 25, 27, 30};
// int CS[num] = {4, 5, 7, 2, 11, 25, 15, 16, 3, 26};
// int CS[num] = {4,2,3,5,7,11,15,16};
// int CS[num] = {0x0C};
// int CS[num] = {0x0C};
// int CS[num] = {0x11,0x0D,0x12,0x10,0x13,0x0E,0x0C,0x0F};
// int CS[num] = {0x0C,0x0D,0x0E};
int CS[num] = {0x0C,0x0D,0x0E,0x0F,0x10,0x11,0x12,0x13};
// int CS[num] = {0x0F};
// int CS[num] = { 2,3, 4, 5, 7,11, 15, 16,25,26};
// int CS[num] = {  5, 7,11, 15, 16,25,26};
// int CS[num] = {2, 3,4,5,7 11, 15, 16, 25,26};
//int CS[num] = {2, 3, 4, 5, 7, 11, 15, 16, 25, 26, 27, 30}; // platform 1
//int CS[num] = {16, 15, 11, A1, A0, 7, 27, 26, 25, A5, A4, 30};
//int CS[num] = {27};
// BLE Service
int start_time_global = 0;
BLEDfu  bledfu;  // OTA DFU service
BLEDis  bledis;  // device information
BLEUart bleuart; // uart over ble
BLEBas  blebas;  // battery
float data_array[num*3+1];   // The array to hold the data
void setup()
{
 NRF_POWER->DCDCEN=0;
  NRF_CLOCK->LFCLKSRC = CLOCK_LFCLKSRC_SRC_RC << CLOCK_LFCLKSRC_SRC_Pos;
  NRF_CLOCK->TASKS_LFCLKSTART = 1;
  // 等待 LFCLK 启动完成
  while (NRF_CLOCK->EVENTS_LFCLKSTARTED == 0);
  NRF_CLOCK->EVENTS_LFCLKSTARTED = 0;  // 清除标志
  // NRF_CLOCK->LFCLKSRC = CLOCK_LFCLKSRC_SRC_RC << CLOCK_LFCLKSRC_SRC_Pos;
  // NRF_CLOCK->TASKS_LFCLKSTART = 1;
  // while (NRF_CLOCK->EVENTS_LFCLKSTARTE)
  // Wire.setClock(100000); // 设置 I2C 时钟为 400kHz
  Wire.setPins(SDA_PIN, SCL_PIN); // 设置 SDA 和 SCL 引脚
  WIRE.begin();
  dwt_enable();         // For more accurate micros() on Feather
  Serial.begin(115200);
#if CFG_DEBUG
  // Blocking wait for connection when debug mode is enabled via IDE
  while ( !Serial ) yield();
#endif
  //********************BLE SETUP*******************//
  Serial.println("Bluefruit52 BLEUART Example");
  Serial.println("---------------------------\n");
  Serial.print("Configured SDA: ");
  Serial.println(SDA_PIN);
  Serial.print("Configured SCL: ");
  Serial.println(SCL_PIN);
  // Setup the BLE LED to be enabled on CONNECT
  // Note: This is actually the default behavior, but provided
  // here in case you want to control this LED manually via PIN 19
  Bluefruit.autoConnLed(true);
  // Config the peripheral connection with maximum bandwidth
  // more SRAM required by SoftDevice
  // Note: All config***() function must be called before begin()
  Bluefruit.configPrphBandwidth(BANDWIDTH_MAX);
  Bluefruit.begin();
  Bluefruit.setTxPower(4);    // Check bluefruit.h for supported values
  Bluefruit.setName("Bluefruit52");
  // setConnectionParams();
  //Bluefruit.setName(getMcuUniqueID()); // useful testing with multiple central connections
  Bluefruit.Periph.setConnectCallback(connect_callback);
  Bluefruit.Periph.setDisconnectCallback(disconnect_callback);
  // To be consistent OTA DFU should be added first if it exists
  bledfu.begin();
  // Configure and Start Device Information Service
  bledis.setManufacturer("Adafruit Industries");
  bledis.setModel("Bluefruit Feather52");
  bledis.begin();
  // Configure and Start BLE Uart Service
  bleuart.begin();
  // Start BLE Battery Service
  blebas.begin();
  blebas.write(100);
  // Set up and start advertising
  startAdv();
  //*****************SENSOR SETUP*****************//
  //注释
  pinMode(LED_BUILTIN, OUTPUT);     // Indicator of whether the sensors are all found
  digitalWrite(LED_BUILTIN, LOW);
  delayMicroseconds(2);
  for(int i = 0; i < num; ++i){
    // SPISettings mySPISettings(10000000, MSBFIRST, SPI_MODE3); // 10MHz, MSB first, Mode 0
    // Wire.begin(newSDA, newSCL); // 指定新的 SDA 和 SCL 引脚
    sensor[i] = Adafruit_MLX90393();
    // 尝试重新初始化传感器的次数
    int attempts = 0;
    const int maxAttempts = 100; // 最多尝试100次
    // while (!sensor[i].begin_I2C(CS[i]),&WIRE) {
    //     Serial.print("No sensor ");
    //     Serial.print(i + 1);
    //     Serial.println(" found ... check your wiring?");
    //     // delay(500);
    //     delayMicroseconds(1000);
    //     attempts++;
    //     if (attempts >= maxAttempts) {
    //         Serial.println("Maximum attempts reached. Skipping this sensor.");
    //         break;
    //     }
    // }
    while (! sensor[i].begin_I2C(CS[i],&WIRE)) {
      Serial.print("No sensor ");
      Serial.print(i+1);
      Serial.println(" found ... check your wiring?");
       delayMicroseconds(500);
    }
    while(!sensor[i].setOversampling(MLX90393_OSR_3)){
      Serial.print("Sensor ");
      Serial.print(i+1);
      Serial.println(" reset OSR!");
      delayMicroseconds(500);
    }
    delayMicroseconds(1000);
    while(!sensor[i].setFilter(MLX90393_FILTER_5)){
      Serial.print("Sensor ");
      Serial.print(i+1);
      Serial.println(" reset filter!");
      delayMicroseconds(500);
    }
  }
  //注释
  digitalWrite(LED_BUILTIN, HIGH);
  Serial.println("Initialization attempt for all MLX90393 sensors completed.");
// }
  // digitalWrite(LED_BUILTIN, LOW);
  // delayMicroseconds(2);
  // for(int i = 0; i < num; ++i){
  //   sensor[i] = Adafruit_MLX90393();
  //   while (! sensor[i].begin_SPI(CS[i])) {
  //     //  Serial.println("No sensor found ... check your wiring?");
  //     //  while (1) { delay(10); }
  //     Serial.print("No sensor ");
  //     Serial.print(i+1);
  //     Serial.println(" found ... check your wiring?");
  //     // delay(100);
  //   }
  //   Serial.print("Sensor ");
  //   Serial.print(i+1);
  //   Serial.println(" found!");
    // while(!sensor[i].setOversampling(MLX90393_OSR_3)){
    //   Serial.print("Sensor ");
    //   Serial.print(i+1);
    //   Serial.println(" reset OSR!");
    //   delayMicroseconds(500);
    // }
    // delayMicroseconds(500);
    // while(!sensor[i].setFilter(MLX90393_FILTER_5)){
    //   Serial.print("Sensor ");
    //   Serial.print(i+1);
    //   Serial.println(" reset filter!");
    //   delayMicroseconds(1000);
    // }
  // }
  // digitalWrite(LED_BUILTIN, HIGH);
  // Serial.println("Found all MLX90393 sensors");
//  bleuart.setRxCallback(rx_callback);/
}
void startAdv(void)
{
  // Advertising packet
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  // Include bleuart 128-bit uuid
  Bluefruit.Advertising.addService(bleuart);
  // Secondary Scan Response packet (optional)
  // Since there is no room for 'Name' in Advertising packet
  Bluefruit.ScanResponse.addName();
  /* Start Advertising
   * - Enable auto advertising if disconnected
   * - Interval:  fast mode = 20 ms, slow mode = 152.5 ms
   * - Timeout for fast mode is 30 seconds
   * - Start(timeout) with timeout = 0 will advertise forever (until connected)
   *
   * For recommended advertising interval
   * https://developer.apple.com/library/content/qa/qa1931/_index.html
   */
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(32, 32);    // in unit of 0.625 ms
  Bluefruit.Advertising.setFastTimeout(30);      // number of seconds in fast mode
  Bluefruit.Advertising.start(0);                // 0 = Don't stop advertising after n seconds
}
void rx_callback(uint16_t conn_hdl) {
  while (bleuart.available()) {
    char ch = bleuart.read();
    // 收集字符串或处理特定命令
    if (ch == 'P') {
      // 简单示例：收到 'P' 表示是 "PONG"
      unsigned long rtt = micros() - start_time_global;
      Serial.print("RTT: ");
      Serial.print(rtt);
      Serial.println(" us");
    }
  }
}


void loop()
{
  // int start_time = micros();
   //Serial.println("###################");
   for(int i = 0; i < num; ++i){
      sensor[i].startSingleMeasurement();
      //delayMicroseconds(50);
   }
   delayMicroseconds(mlx90393_tconv[5][3]*1000);
   for(int i = 0; i < num; ++i){
      if(!sensor[i].readMeasurement(&data_array[3*i], &data_array[3*i+1], &data_array[3*i+2])){
        Serial.print("Sensor ");
        Serial.print(i+1);
        Serial.println(" no data read!");
        //注释
        digitalWrite(LED_BUILTIN, LOW);
      }
      // else{
      // Serial.print(i+1); Serial.print(" ");Serial.print(data_array[3*i], 4); Serial.print(" uT");
      //  Serial.print(data_array[3*i+1], 4); Serial.print(" uT");
      // Serial.print(data_array[3*i+2], 4); Serial.println(" uT");
      // }
      // delayMicroseconds(700);
    }
    // delayMicroseconds(10000);
    // Measure battery voltage
    /*
    float measuredvbat = analogRead(A7);
    measuredvbat *= 2;    // we divided by 2, so multiply back
    measuredvbat *= 3.3;  // Multiply by 3.3V, our reference voltage
    measuredvbat /= 1024; // convert to voltage
    data_array[num*3] = measuredvbat;*/
    // Send to PC
    int start_time = micros();
    uint32_t t = micros();  // 记录时间戳
    memcpy((void*)&data_array[num*3], &t, sizeof(t));  // 附加到数组尾部
    bleuart.write((byte*)(data_array), num*3*4 + 4);  // 原数据 + 时间戳
//    bleuart.write((byte*)(data_array), num*3*/4+4);
    int elapsed_time = micros() - start_time;
    Serial.print(1000000 / elapsed_time);
    Serial.println(" Hz");
//    start_time_global = micros();
//    bleuart.write((uint8_t *)"PING", 4);
////    unsigned long t = micros();
////    bleuart.write((uint8_t*)&t, sizeof(t));  // 发送4字节时间戳
//    delay(500); // 等待 PC 回复 "PONG"
}
// callback invoked when central connects
void connect_callback(uint16_t conn_handle)
{
  // Get the reference to current connection
  BLEConnection* connection = Bluefruit.Connection(conn_handle);
  char central_name[32] = { 0 };
  connection->getPeerName(central_name, sizeof(central_name));
  Serial.print("Connected to ");
  Serial.println(central_name);
}
/**
 * Callback invoked when a connection is dropped
 * @param conn_handle connection where this event happens
 * @param reason is a BLE_HCI_STATUS_CODE which can be found in ble_hci.h
 */
void disconnect_callback(uint16_t conn_handle, uint8_t reason)
{
  (void) conn_handle;
  (void) reason;
  Serial.println();
  Serial.print("Disconnected, reason = 0x"); Serial.println(reason, HEX);
}
