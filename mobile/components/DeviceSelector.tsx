import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Modal,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  Alert,
  TextInput,
  Platform,
} from 'react-native';
import Svg, { Path, Circle, Rect } from 'react-native-svg';
import colors from '../constants/colors';
import { Button } from './ui/Button';
import { Card } from './ui/Card';
import { ConnectedDevice } from '../store/deviceStore';
import {
  deviceDatabase,
  getAllManufacturers,
  getDevicesByManufacturer,
  getDeviceFeatureDescription,
  getConnectivityDescription,
  findDeviceByBluetoothName,
  DeviceModel,
  Manufacturer,
} from '../data/deviceDatabase';

// Icons
function PlusIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M12 5v14M5 12h14" />
    </Svg>
  );
}

function BluetoothIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M6.5 6.5l11 11L12 23V1l5.5 5.5-11 11" />
    </Svg>
  );
}

function WifiIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M5 12.55a11 11 0 0 1 14.08 0M1.42 9a16 16 0 0 1 21.16 0M8.53 16.11a6 6 0 0 1 6.95 0M12 20h.01" />
    </Svg>
  );
}

function CheckIcon({ color = colors.status.success, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M20 6L9 17l-5-5" />
    </Svg>
  );
}

function ChevronRightIcon({ color = colors.text.muted, size = 20 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M9 18l6-6-6-6" />
    </Svg>
  );
}

function ChevronLeftIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M15 18l-6-6 6-6" />
    </Svg>
  );
}

function CloseIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M18 6L6 18M6 6l12 12" />
    </Svg>
  );
}

function SearchIcon({ color = colors.text.muted, size = 20 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Circle cx={11} cy={11} r={8} />
      <Path d="M21 21l-4.35-4.35" />
    </Svg>
  );
}

interface DeviceSelectorProps {
  visible: boolean;
  onClose: () => void;
  onDeviceConnected: (device: ConnectedDevice) => void;
  connectedDevice?: ConnectedDevice | null;
}

type Step = 'main' | 'manufacturer' | 'model' | 'connecting' | 'handy';

export function DeviceSelector({ visible, onClose, onDeviceConnected, connectedDevice }: DeviceSelectorProps) {
  const [step, setStep] = useState<Step>('main');
  const [selectedManufacturer, setSelectedManufacturer] = useState<Manufacturer | null>(null);
  const [selectedDevice, setSelectedDevice] = useState<DeviceModel | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [handyKey, setHandyKey] = useState('');
  const [discoveredDevices, setDiscoveredDevices] = useState<Array<{ id: string; name: string }>>([]);

  const manufacturers = getAllManufacturers();

  const filteredManufacturers = searchQuery
    ? manufacturers.filter(m =>
        m.displayName.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : manufacturers;

  const resetState = useCallback(() => {
    setStep('main');
    setSelectedManufacturer(null);
    setSelectedDevice(null);
    setSearchQuery('');
    setHandyKey('');
    setIsConnecting(false);
    setIsScanning(false);
    setDiscoveredDevices([]);
  }, []);

  const handleClose = () => {
    resetState();
    onClose();
  };

  const handleSelectManufacturer = (manufacturerName: string) => {
    const manufacturer = deviceDatabase.find(m => m.name === manufacturerName);
    if (manufacturer) {
      if (manufacturer.name === 'thehandy') {
        // Special case for The Handy - needs connection key
        setSelectedManufacturer(manufacturer);
        setStep('handy');
      } else {
        setSelectedManufacturer(manufacturer);
        setStep('model');
      }
    }
  };

  const handleSelectDevice = async (device: DeviceModel) => {
    setSelectedDevice(device);
    setStep('connecting');
    setIsScanning(true);

    // Start Bluetooth scanning
    await scanForDevice(device);
  };

  const scanForDevice = async (device: DeviceModel) => {
    try {
      // Check if Web Bluetooth is supported (for web/expo-web)
      if (Platform.OS === 'web' && typeof navigator !== 'undefined' && 'bluetooth' in navigator) {
        const btDevice = await (navigator as any).bluetooth.requestDevice({
          filters: device.bluetoothNames.map(name => ({ namePrefix: name.substring(0, 5) })),
          optionalServices: ['battery_service', '0000fff0-0000-1000-8000-00805f9b34fb'],
        });

        if (btDevice) {
          setDiscoveredDevices([{ id: btDevice.id, name: btDevice.name || device.name }]);
          await connectToDevice(btDevice.id, btDevice.name || device.name, device);
        }
      } else {
        // For native platforms, simulate device discovery
        // In production, this would use react-native-ble-plx or similar
        setIsScanning(true);

        // Simulate scanning delay
        setTimeout(() => {
          setIsScanning(false);
          // For demo, show the device as "found"
          setDiscoveredDevices([{ id: `sim-${Date.now()}`, name: device.name }]);
        }, 2000);
      }
    } catch (error: any) {
      setIsScanning(false);
      if (error.message?.includes('cancelled')) {
        // User cancelled - go back to model selection
        setStep('model');
      } else {
        Alert.alert(
          'Scanning Failed',
          'Could not scan for Bluetooth devices. Make sure Bluetooth is enabled and try again.',
          [{ text: 'OK', onPress: () => setStep('model') }]
        );
      }
    }
  };

  const connectToDevice = async (deviceId: string, deviceName: string, device: DeviceModel) => {
    setIsConnecting(true);
    try {
      // Simulate connection delay
      await new Promise(resolve => setTimeout(resolve, 1500));

      const connectedDevice: ConnectedDevice = {
        id: deviceId,
        name: deviceName,
        manufacturer: selectedManufacturer?.displayName || 'Unknown',
        model: device.name,
        type: 'bluetooth',
        features: device.features,
        connectivity: device.connectivity,
        batteryLevel: Math.floor(Math.random() * 40) + 60, // Simulated battery level
        status: 'connected',
      };

      onDeviceConnected(connectedDevice);
      handleClose();

      Alert.alert(
        'Device Connected',
        `Successfully connected to ${deviceName}!`,
        [{ text: 'OK' }]
      );
    } catch (error) {
      Alert.alert(
        'Connection Failed',
        'Could not connect to the device. Please try again.',
        [{ text: 'OK' }]
      );
    } finally {
      setIsConnecting(false);
    }
  };

  const handleConnectHandy = async () => {
    if (!handyKey.trim()) {
      Alert.alert('Connection Key Required', 'Please enter your Handy connection key.');
      return;
    }

    setIsConnecting(true);
    try {
      // Verify the connection key with The Handy API
      const response = await fetch(`https://www.handyfeeling.com/api/handy/v2/connected`, {
        headers: {
          'X-Connection-Key': handyKey.trim(),
        },
      });

      if (response.ok) {
        const data = await response.json();
        if (data.connected) {
          const handyDevice = deviceDatabase.find(m => m.name === 'thehandy')?.devices[0];
          const connectedDevice: ConnectedDevice = {
            id: handyKey.trim(),
            name: 'The Handy',
            manufacturer: 'The Handy',
            model: 'The Handy',
            type: 'handy',
            features: handyDevice?.features || ['linear'],
            connectivity: handyDevice?.connectivity || ['WiFi'],
            status: 'connected',
          };

          onDeviceConnected(connectedDevice);
          handleClose();

          Alert.alert(
            'Device Connected',
            'Successfully connected to The Handy!',
            [{ text: 'OK' }]
          );
        } else {
          throw new Error('Device not connected');
        }
      } else {
        throw new Error('Invalid connection key');
      }
    } catch (error) {
      Alert.alert(
        'Connection Failed',
        'Could not connect to The Handy. Please check your connection key and make sure the device is online.',
        [{ text: 'OK' }]
      );
    } finally {
      setIsConnecting(false);
    }
  };

  const renderHeader = () => {
    const titles: Record<Step, string> = {
      main: 'Add Device',
      manufacturer: 'Select Brand',
      model: selectedManufacturer?.displayName || 'Select Model',
      connecting: 'Connecting',
      handy: 'The Handy',
    };

    const showBack = step !== 'main';

    return (
      <View style={styles.header}>
        {showBack ? (
          <TouchableOpacity
            onPress={() => {
              if (step === 'model' || step === 'handy') setStep('manufacturer');
              else if (step === 'connecting') setStep('model');
              else if (step === 'manufacturer') setStep('main');
            }}
            style={styles.headerButton}
          >
            <ChevronLeftIcon />
          </TouchableOpacity>
        ) : (
          <View style={styles.headerButton} />
        )}
        <Text style={styles.headerTitle}>{titles[step]}</Text>
        <TouchableOpacity onPress={handleClose} style={styles.headerButton}>
          <CloseIcon />
        </TouchableOpacity>
      </View>
    );
  };

  const renderMainStep = () => (
    <View style={styles.stepContainer}>
      <View style={styles.iconContainer}>
        <BluetoothIcon color={colors.primary.DEFAULT} size={48} />
      </View>
      <Text style={styles.stepTitle}>Connect Your Device</Text>
      <Text style={styles.stepDescription}>
        FunGen supports 200+ devices from major brands including Lovense, Kiiroo, We-Vibe, Satisfyer, and more.
      </Text>

      <View style={styles.buttonContainer}>
        <Button
          title="Add Your Device"
          onPress={() => setStep('manufacturer')}
          fullWidth
          icon={<PlusIcon color="#FFFFFF" size={20} />}
        />

        {connectedDevice && (
          <Card style={styles.connectedDeviceCard}>
            <View style={styles.connectedDeviceRow}>
              <View style={styles.connectedDeviceInfo}>
                <Text style={styles.connectedDeviceName}>{connectedDevice.name}</Text>
                <Text style={styles.connectedDeviceStatus}>
                  {connectedDevice.status === 'connected' ? 'Connected' : connectedDevice.status}
                </Text>
              </View>
              <CheckIcon />
            </View>
          </Card>
        )}
      </View>
    </View>
  );

  const renderManufacturerStep = () => (
    <View style={styles.stepContainer}>
      <View style={styles.searchContainer}>
        <SearchIcon />
        <TextInput
          style={styles.searchInput}
          placeholder="Search brands..."
          placeholderTextColor={colors.text.muted}
          value={searchQuery}
          onChangeText={setSearchQuery}
          autoCapitalize="none"
        />
      </View>

      <ScrollView style={styles.listContainer} showsVerticalScrollIndicator={false}>
        {filteredManufacturers.map((manufacturer) => (
          <TouchableOpacity
            key={manufacturer.name}
            style={styles.listItem}
            onPress={() => handleSelectManufacturer(manufacturer.name)}
            activeOpacity={0.7}
          >
            <Text style={styles.listItemText}>{manufacturer.displayName}</Text>
            <ChevronRightIcon />
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );

  const renderModelStep = () => {
    const devices = selectedManufacturer ? getDevicesByManufacturer(selectedManufacturer.name) : [];

    return (
      <View style={styles.stepContainer}>
        <ScrollView style={styles.listContainer} showsVerticalScrollIndicator={false}>
          {devices.map((device, index) => (
            <TouchableOpacity
              key={`${device.name}-${index}`}
              style={styles.deviceItem}
              onPress={() => handleSelectDevice(device)}
              activeOpacity={0.7}
            >
              <View style={styles.deviceItemContent}>
                <Text style={styles.deviceItemName}>{device.name}</Text>
                <Text style={styles.deviceItemFeatures}>
                  {getDeviceFeatureDescription(device.features)}
                </Text>
                <View style={styles.deviceItemTags}>
                  {device.connectivity.map((conn) => (
                    <View key={conn} style={styles.tag}>
                      {conn === 'BT4LE' ? (
                        <BluetoothIcon color={colors.primary.DEFAULT} size={12} />
                      ) : (
                        <WifiIcon color={colors.primary.DEFAULT} size={12} />
                      )}
                      <Text style={styles.tagText}>
                        {conn === 'BT4LE' ? 'Bluetooth' : conn}
                      </Text>
                    </View>
                  ))}
                </View>
              </View>
              <ChevronRightIcon />
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>
    );
  };

  const renderConnectingStep = () => (
    <View style={styles.stepContainer}>
      <View style={styles.connectingContainer}>
        {isScanning ? (
          <>
            <ActivityIndicator size="large" color={colors.primary.DEFAULT} />
            <Text style={styles.connectingTitle}>Scanning for {selectedDevice?.name}...</Text>
            <Text style={styles.connectingDescription}>
              Make sure your device is turned on and in pairing mode.
            </Text>
          </>
        ) : isConnecting ? (
          <>
            <ActivityIndicator size="large" color={colors.primary.DEFAULT} />
            <Text style={styles.connectingTitle}>Connecting...</Text>
            <Text style={styles.connectingDescription}>
              Establishing connection with your device.
            </Text>
          </>
        ) : discoveredDevices.length > 0 ? (
          <>
            <Text style={styles.connectingTitle}>Device Found</Text>
            <Text style={styles.connectingDescription}>
              Tap to connect to your device.
            </Text>
            {discoveredDevices.map((device) => (
              <TouchableOpacity
                key={device.id}
                style={styles.discoveredDevice}
                onPress={() => connectToDevice(device.id, device.name, selectedDevice!)}
              >
                <BluetoothIcon color={colors.primary.DEFAULT} size={24} />
                <Text style={styles.discoveredDeviceName}>{device.name}</Text>
                <ChevronRightIcon />
              </TouchableOpacity>
            ))}
          </>
        ) : (
          <>
            <Text style={styles.connectingTitle}>No Devices Found</Text>
            <Text style={styles.connectingDescription}>
              Make sure your device is turned on and in pairing mode, then try again.
            </Text>
            <Button
              title="Scan Again"
              onPress={() => selectedDevice && scanForDevice(selectedDevice)}
              style={{ marginTop: 16 }}
            />
          </>
        )}
      </View>
    </View>
  );

  const renderHandyStep = () => (
    <View style={styles.stepContainer}>
      <View style={styles.handyContainer}>
        <View style={styles.iconContainer}>
          <WifiIcon color={colors.primary.DEFAULT} size={48} />
        </View>
        <Text style={styles.stepTitle}>Connect via WiFi</Text>
        <Text style={styles.stepDescription}>
          The Handy connects via WiFi using a connection key. You can find your key in the Handy app or on handyfeeling.com
        </Text>

        <View style={styles.inputContainer}>
          <Text style={styles.inputLabel}>Connection Key</Text>
          <TextInput
            style={styles.textInput}
            placeholder="Enter your connection key"
            placeholderTextColor={colors.text.muted}
            value={handyKey}
            onChangeText={setHandyKey}
            autoCapitalize="characters"
            autoCorrect={false}
          />
        </View>

        <Button
          title={isConnecting ? 'Connecting...' : 'Connect'}
          onPress={handleConnectHandy}
          loading={isConnecting}
          disabled={!handyKey.trim() || isConnecting}
          fullWidth
        />
      </View>
    </View>
  );

  const renderStep = () => {
    switch (step) {
      case 'main':
        return renderMainStep();
      case 'manufacturer':
        return renderManufacturerStep();
      case 'model':
        return renderModelStep();
      case 'connecting':
        return renderConnectingStep();
      case 'handy':
        return renderHandyStep();
    }
  };

  return (
    <Modal
      visible={visible}
      animationType="slide"
      presentationStyle="pageSheet"
      onRequestClose={handleClose}
    >
      <View style={styles.container}>
        {renderHeader()}
        {renderStep()}
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg.base,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.DEFAULT,
  },
  headerButton: {
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: colors.text.primary,
  },
  stepContainer: {
    flex: 1,
    padding: 16,
  },
  iconContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: colors.primary.dim,
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'center',
    marginBottom: 24,
    marginTop: 24,
  },
  stepTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: colors.text.primary,
    textAlign: 'center',
    marginBottom: 12,
  },
  stepDescription: {
    fontSize: 16,
    color: colors.text.secondary,
    textAlign: 'center',
    marginBottom: 32,
    lineHeight: 24,
  },
  buttonContainer: {
    gap: 16,
  },
  connectedDeviceCard: {
    marginTop: 8,
  },
  connectedDeviceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 12,
  },
  connectedDeviceInfo: {
    flex: 1,
  },
  connectedDeviceName: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.primary,
  },
  connectedDeviceStatus: {
    fontSize: 14,
    color: colors.status.success,
    marginTop: 2,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.bg.elevated,
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    marginBottom: 16,
    gap: 12,
  },
  searchInput: {
    flex: 1,
    fontSize: 16,
    color: colors.text.primary,
  },
  listContainer: {
    flex: 1,
  },
  listItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 16,
    paddingHorizontal: 4,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.DEFAULT,
  },
  listItemText: {
    fontSize: 16,
    color: colors.text.primary,
    fontWeight: '500',
  },
  deviceItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 16,
    paddingHorizontal: 4,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.DEFAULT,
  },
  deviceItemContent: {
    flex: 1,
  },
  deviceItemName: {
    fontSize: 16,
    color: colors.text.primary,
    fontWeight: '600',
    marginBottom: 4,
  },
  deviceItemFeatures: {
    fontSize: 14,
    color: colors.text.secondary,
    marginBottom: 8,
  },
  deviceItemTags: {
    flexDirection: 'row',
    gap: 8,
  },
  tag: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.primary.dim,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    gap: 4,
  },
  tagText: {
    fontSize: 12,
    color: colors.primary.DEFAULT,
    fontWeight: '500',
  },
  connectingContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 32,
  },
  connectingTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: colors.text.primary,
    marginTop: 24,
    marginBottom: 8,
    textAlign: 'center',
  },
  connectingDescription: {
    fontSize: 16,
    color: colors.text.secondary,
    textAlign: 'center',
    lineHeight: 24,
  },
  discoveredDevice: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.bg.elevated,
    borderRadius: 12,
    padding: 16,
    marginTop: 24,
    width: '100%',
    gap: 12,
  },
  discoveredDeviceName: {
    flex: 1,
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.primary,
  },
  handyContainer: {
    flex: 1,
    paddingTop: 24,
  },
  inputContainer: {
    marginBottom: 24,
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text.secondary,
    marginBottom: 8,
  },
  textInput: {
    backgroundColor: colors.bg.elevated,
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 14,
    fontSize: 16,
    color: colors.text.primary,
    borderWidth: 1,
    borderColor: colors.border.DEFAULT,
  },
});
