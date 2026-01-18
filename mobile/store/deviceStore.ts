import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { DeviceModel } from '../data/deviceDatabase';

export interface ConnectedDevice {
  id: string;
  name: string;
  manufacturer: string;
  model: string;
  type: 'bluetooth' | 'wifi' | 'handy';
  features: DeviceModel['features'];
  connectivity: DeviceModel['connectivity'];
  batteryLevel?: number;
  status: 'connected' | 'connecting' | 'disconnected' | 'error';
  connectionKey?: string; // For Handy
}

interface DeviceState {
  connectedDevice: ConnectedDevice | null;
  isConnecting: boolean;
  lastError: string | null;

  // Actions
  setConnectedDevice: (device: ConnectedDevice | null) => void;
  setIsConnecting: (connecting: boolean) => void;
  setLastError: (error: string | null) => void;
  disconnect: () => void;

  // Device commands
  sendCommand: (command: DeviceCommand) => Promise<void>;
  setPosition: (position: number) => Promise<void>;
  setSpeed: (speed: number) => Promise<void>;
  vibrate: (intensity: number) => Promise<void>;
  stop: () => Promise<void>;
}

export type DeviceCommand =
  | { type: 'setPosition'; position: number; speed?: number }
  | { type: 'setSpeed'; speed: number }
  | { type: 'vibrate'; intensity: number }
  | { type: 'stop' }
  | { type: 'sync'; timestamps: number[]; positions: number[] };

export const useDeviceStore = create<DeviceState>()(
  persist(
    (set, get) => ({
      connectedDevice: null,
      isConnecting: false,
      lastError: null,

      setConnectedDevice: (device) => set({ connectedDevice: device, lastError: null }),

      setIsConnecting: (connecting) => set({ isConnecting: connecting }),

      setLastError: (error) => set({ lastError: error }),

      disconnect: () => {
        set({ connectedDevice: null, lastError: null });
      },

      sendCommand: async (command: DeviceCommand) => {
        const { connectedDevice } = get();
        if (!connectedDevice) {
          throw new Error('No device connected');
        }

        try {
          switch (connectedDevice.type) {
            case 'handy':
              await sendHandyCommand(connectedDevice, command);
              break;
            case 'bluetooth':
              await sendBluetoothCommand(connectedDevice, command);
              break;
            default:
              console.log('Device command:', command);
          }
        } catch (error: any) {
          set({ lastError: error.message });
          throw error;
        }
      },

      setPosition: async (position: number) => {
        await get().sendCommand({ type: 'setPosition', position });
      },

      setSpeed: async (speed: number) => {
        await get().sendCommand({ type: 'setSpeed', speed });
      },

      vibrate: async (intensity: number) => {
        await get().sendCommand({ type: 'vibrate', intensity });
      },

      stop: async () => {
        await get().sendCommand({ type: 'stop' });
      },
    }),
    {
      name: 'fungen-device-storage',
      storage: createJSONStorage(() => AsyncStorage),
      partialize: (state) => ({
        // Only persist device info, not connection state
        connectedDevice: state.connectedDevice
          ? { ...state.connectedDevice, status: 'disconnected' }
          : null,
      }),
    }
  )
);

// Handy API integration
const HANDY_API_BASE = 'https://www.handyfeeling.com/api/handy/v2';

async function sendHandyCommand(device: ConnectedDevice, command: DeviceCommand): Promise<void> {
  const connectionKey = device.connectionKey || device.id;

  const headers = {
    'X-Connection-Key': connectionKey,
    'Content-Type': 'application/json',
  };

  switch (command.type) {
    case 'setPosition': {
      // Set HDSP (Handy Direct Streaming Protocol) position
      await fetch(`${HANDY_API_BASE}/hdsp/xat`, {
        method: 'PUT',
        headers,
        body: JSON.stringify({
          stopOnTarget: true,
          immediateResponse: true,
          position: Math.round(command.position * 100), // 0-100 scale
          velocity: command.speed ? Math.round(command.speed * 400) : 400, // mm/s
        }),
      });
      break;
    }

    case 'setSpeed': {
      // Set slide settings
      await fetch(`${HANDY_API_BASE}/slide`, {
        method: 'PUT',
        headers,
        body: JSON.stringify({
          min: 0,
          max: 100,
        }),
      });
      break;
    }

    case 'sync': {
      // Prepare for sync playback
      // This would typically upload a funscript and start sync mode
      await fetch(`${HANDY_API_BASE}/hssp/setup`, {
        method: 'PUT',
        headers,
        body: JSON.stringify({
          serverTime: Date.now(),
        }),
      });
      break;
    }

    case 'stop': {
      await fetch(`${HANDY_API_BASE}/hdsp/xat`, {
        method: 'PUT',
        headers,
        body: JSON.stringify({
          stopOnTarget: true,
          immediateResponse: true,
          position: 0,
          velocity: 100,
        }),
      });
      break;
    }
  }
}

// Bluetooth device command sending (uses Web Bluetooth or BLE library)
async function sendBluetoothCommand(device: ConnectedDevice, command: DeviceCommand): Promise<void> {
  // For Bluetooth devices, we need to send commands via the appropriate protocol
  // This is a simplified implementation - in production, you'd use buttplug.io
  // or direct BLE characteristic writes based on the device manufacturer

  console.log(`Sending command to ${device.name}:`, command);

  // Different devices have different protocols
  // Lovense uses their own protocol, Kiiroo uses another, etc.
  // The buttplug.io library handles all of this abstraction

  switch (command.type) {
    case 'vibrate': {
      // Most vibrating devices accept a 0-1 intensity value
      // Convert to device-specific protocol
      console.log(`Vibrate: ${command.intensity * 100}%`);
      break;
    }

    case 'setPosition': {
      // Linear devices (strokers) accept position commands
      console.log(`Position: ${command.position * 100}%`);
      break;
    }

    case 'stop': {
      console.log('Stop all actuators');
      break;
    }
  }
}

// Export helper to sync funscript data to device
export async function syncFunscriptToDevice(
  timestamps: number[],
  positions: number[],
  currentTime: number
): Promise<void> {
  const store = useDeviceStore.getState();
  const device = store.connectedDevice;

  if (!device) return;

  // Find the current and next point based on current playback time
  let currentIndex = -1;
  for (let i = 0; i < timestamps.length - 1; i++) {
    if (timestamps[i] <= currentTime && timestamps[i + 1] > currentTime) {
      currentIndex = i;
      break;
    }
  }

  if (currentIndex === -1) return;

  const currentPos = positions[currentIndex];
  const nextPos = positions[currentIndex + 1];
  const currentTs = timestamps[currentIndex];
  const nextTs = timestamps[currentIndex + 1];

  // Interpolate position based on current time
  const progress = (currentTime - currentTs) / (nextTs - currentTs);
  const interpolatedPos = currentPos + (nextPos - currentPos) * progress;

  // Calculate speed needed to reach next position
  const timeDelta = (nextTs - currentTime) / 1000; // seconds
  const posDelta = Math.abs(nextPos - interpolatedPos);
  const speed = timeDelta > 0 ? posDelta / timeDelta : 1;

  // Send position command
  await store.sendCommand({
    type: 'setPosition',
    position: interpolatedPos / 100, // Normalize to 0-1
    speed: Math.min(speed, 1),
  });
}
