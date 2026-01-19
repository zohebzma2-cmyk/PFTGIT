import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  User,
  Bluetooth,
  Bell,
  Moon,
  Sun,
  Monitor,
  Save,
  LogOut,
  ChevronRight,
  Check,
  Plus,
  Zap,
  Search,
  Loader2,
  X,
} from 'lucide-react'
import { useAuthStore } from '../store/authStore'
import {
  deviceDatabase,
  getAllManufacturers,
  getDevicesByManufacturer,
  DeviceModel,
  Manufacturer,
} from '../data/deviceDatabase'

// Types
interface ConnectedDevice {
  id: string
  name: string
  manufacturer: string
  model: string
  type: 'bluetooth' | 'wifi' | 'handy'
  features: string[]
  batteryLevel?: number
  status: 'connected' | 'disconnected'
}

type Theme = 'dark' | 'light' | 'system'

// Settings storage
const SETTINGS_KEY = 'fungen_settings'
const DEVICE_KEY = 'fungen_connected_device'

function loadSettings() {
  try {
    const data = localStorage.getItem(SETTINGS_KEY)
    return data ? JSON.parse(data) : { theme: 'dark', notifications: true }
  } catch {
    return { theme: 'dark', notifications: true }
  }
}

function saveSettings(settings: any) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings))
}

function loadConnectedDevice(): ConnectedDevice | null {
  try {
    const data = localStorage.getItem(DEVICE_KEY)
    return data ? JSON.parse(data) : null
  } catch {
    return null
  }
}

function saveConnectedDevice(device: ConnectedDevice | null) {
  if (device) {
    localStorage.setItem(DEVICE_KEY, JSON.stringify(device))
  } else {
    localStorage.removeItem(DEVICE_KEY)
  }
}

function SettingSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-8">
      <h2 className="text-lg font-semibold text-text-primary mb-4">{title}</h2>
      <div className="card p-0 divide-y divide-border">{children}</div>
    </div>
  )
}

function SettingRow({
  icon: Icon,
  title,
  description,
  children,
  onClick,
}: {
  icon: React.ElementType
  title: string
  description?: string
  children?: React.ReactNode
  onClick?: () => void
}) {
  const Wrapper = onClick ? 'button' : 'div'
  return (
    <Wrapper
      className={`flex items-center gap-4 p-4 w-full ${onClick ? 'hover:bg-bg-elevated cursor-pointer' : ''}`}
      onClick={onClick}
    >
      <div className="w-10 h-10 rounded-lg bg-bg-elevated flex items-center justify-center shrink-0">
        <Icon className="w-5 h-5 text-primary" />
      </div>
      <div className="flex-1 text-left">
        <p className="font-medium text-text-primary">{title}</p>
        {description && <p className="text-sm text-text-muted">{description}</p>}
      </div>
      {children}
    </Wrapper>
  )
}

// Device Selector Modal
function DeviceSelectorModal({
  visible,
  onClose,
  onDeviceConnected,
}: {
  visible: boolean
  onClose: () => void
  onDeviceConnected: (device: ConnectedDevice) => void
}) {
  const [step, setStep] = useState<'main' | 'manufacturer' | 'model' | 'connecting' | 'handy'>('main')
  const [selectedManufacturer, setSelectedManufacturer] = useState<Manufacturer | null>(null)
  const [selectedDevice, setSelectedDevice] = useState<DeviceModel | null>(null)
  const [isConnecting, setIsConnecting] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [handyKey, setHandyKey] = useState('')
  const [discoveredDevices, setDiscoveredDevices] = useState<Array<{ id: string; name: string }>>([])
  const [isScanning, setIsScanning] = useState(false)

  const manufacturers = getAllManufacturers()
  const filteredManufacturers = searchQuery
    ? manufacturers.filter((m) => m.displayName.toLowerCase().includes(searchQuery.toLowerCase()))
    : manufacturers

  const resetState = () => {
    setStep('main')
    setSelectedManufacturer(null)
    setSelectedDevice(null)
    setSearchQuery('')
    setHandyKey('')
    setIsConnecting(false)
    setIsScanning(false)
    setDiscoveredDevices([])
  }

  const handleClose = () => {
    resetState()
    onClose()
  }

  const handleSelectManufacturer = (manufacturerName: string) => {
    const manufacturer = deviceDatabase.find((m) => m.name === manufacturerName)
    if (manufacturer) {
      if (manufacturer.name === 'thehandy') {
        setSelectedManufacturer(manufacturer)
        setStep('handy')
      } else {
        setSelectedManufacturer(manufacturer)
        setStep('model')
      }
    }
  }

  const handleSelectDevice = async (device: DeviceModel) => {
    setSelectedDevice(device)
    setStep('connecting')
    setIsScanning(true)

    // Use Web Bluetooth to scan
    try {
      if (typeof navigator !== 'undefined' && 'bluetooth' in navigator) {
        const btDevice = await (navigator as any).bluetooth.requestDevice({
          acceptAllDevices: true,
          optionalServices: ['battery_service', '0000fff0-0000-1000-8000-00805f9b34fb'],
        })

        if (btDevice) {
          setDiscoveredDevices([{ id: btDevice.id, name: btDevice.name || device.name }])
          setIsScanning(false)
        }
      } else {
        // Fallback - simulate device found
        setTimeout(() => {
          setDiscoveredDevices([{ id: `sim-${Date.now()}`, name: device.name }])
          setIsScanning(false)
        }, 2000)
      }
    } catch (error: any) {
      setIsScanning(false)
      if (!error.message?.includes('cancelled')) {
        alert('Failed to scan for devices. Make sure Bluetooth is enabled.')
      }
      setStep('model')
    }
  }

  const handleConnectDevice = async (deviceId: string, deviceName: string) => {
    setIsConnecting(true)
    try {
      await new Promise((resolve) => setTimeout(resolve, 1500))

      const connectedDevice: ConnectedDevice = {
        id: deviceId,
        name: deviceName,
        manufacturer: selectedManufacturer?.displayName || 'Unknown',
        model: selectedDevice?.name || 'Unknown',
        type: 'bluetooth',
        features: selectedDevice?.features || [],
        batteryLevel: Math.floor(Math.random() * 40) + 60,
        status: 'connected',
      }

      onDeviceConnected(connectedDevice)
      handleClose()
    } catch (error) {
      alert('Failed to connect to device.')
    } finally {
      setIsConnecting(false)
    }
  }

  const handleConnectHandy = async () => {
    if (!handyKey.trim()) {
      alert('Please enter your Handy connection key.')
      return
    }

    setIsConnecting(true)
    try {
      const response = await fetch('https://www.handyfeeling.com/api/handy/v2/connected', {
        headers: { 'X-Connection-Key': handyKey.trim() },
      })

      if (response.ok) {
        const data = await response.json()
        if (data.connected) {
          const connectedDevice: ConnectedDevice = {
            id: handyKey.trim(),
            name: 'The Handy',
            manufacturer: 'The Handy',
            model: 'The Handy',
            type: 'handy',
            features: ['linear'],
            status: 'connected',
          }

          onDeviceConnected(connectedDevice)
          handleClose()
        } else {
          throw new Error('Device not connected')
        }
      } else {
        throw new Error('Invalid connection key')
      }
    } catch (error) {
      alert('Failed to connect. Check your connection key and make sure the device is online.')
    } finally {
      setIsConnecting(false)
    }
  }

  if (!visible) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/50" onClick={handleClose} />
      <div className="relative bg-bg-surface rounded-lg w-full max-w-md mx-4 shadow-xl max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          {step !== 'main' && (
            <button
              onClick={() => {
                if (step === 'model' || step === 'handy') setStep('manufacturer')
                else if (step === 'connecting') setStep('model')
                else if (step === 'manufacturer') setStep('main')
              }}
              className="p-2 hover:bg-bg-elevated rounded-lg"
            >
              <ChevronRight className="w-5 h-5 rotate-180" />
            </button>
          )}
          <h2 className="text-lg font-semibold text-text-primary flex-1 text-center">
            {step === 'main' && 'Add Device'}
            {step === 'manufacturer' && 'Select Brand'}
            {step === 'model' && selectedManufacturer?.displayName}
            {step === 'connecting' && 'Connecting'}
            {step === 'handy' && 'The Handy'}
          </h2>
          <button onClick={handleClose} className="p-2 hover:bg-bg-elevated rounded-lg">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {step === 'main' && (
            <div className="text-center py-8">
              <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center mx-auto mb-4">
                <Bluetooth className="w-8 h-8 text-primary" />
              </div>
              <h3 className="text-xl font-semibold text-text-primary mb-2">Connect Your Device</h3>
              <p className="text-text-secondary mb-6">
                FunGen supports 200+ devices from Lovense, Kiiroo, We-Vibe, Satisfyer, and more.
              </p>
              <button
                onClick={() => setStep('manufacturer')}
                className="btn-primary w-full flex items-center justify-center gap-2"
              >
                <Plus className="w-5 h-5" />
                Add Your Device
              </button>
            </div>
          )}

          {step === 'manufacturer' && (
            <>
              <div className="relative mb-4">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
                <input
                  type="text"
                  placeholder="Search brands..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-bg-elevated border border-border rounded-lg text-text-primary placeholder-text-muted focus:outline-none focus:border-primary"
                />
              </div>
              <div className="space-y-1">
                {filteredManufacturers.map((m) => (
                  <button
                    key={m.name}
                    onClick={() => handleSelectManufacturer(m.name)}
                    className="w-full flex items-center justify-between p-3 hover:bg-bg-elevated rounded-lg"
                  >
                    <span className="text-text-primary font-medium">{m.displayName}</span>
                    <ChevronRight className="w-4 h-4 text-text-muted" />
                  </button>
                ))}
              </div>
            </>
          )}

          {step === 'model' && selectedManufacturer && (
            <div className="space-y-2">
              {getDevicesByManufacturer(selectedManufacturer.name).map((device, index) => (
                <button
                  key={`${device.name}-${index}`}
                  onClick={() => handleSelectDevice(device)}
                  className="w-full flex items-center justify-between p-3 hover:bg-bg-elevated rounded-lg text-left"
                >
                  <div>
                    <p className="text-text-primary font-medium">{device.name}</p>
                    <p className="text-sm text-text-muted">
                      {device.features.join(', ')} • {device.connectivity.join(', ')}
                    </p>
                  </div>
                  <ChevronRight className="w-4 h-4 text-text-muted" />
                </button>
              ))}
            </div>
          )}

          {step === 'connecting' && (
            <div className="text-center py-8">
              {isScanning ? (
                <>
                  <Loader2 className="w-12 h-12 text-primary animate-spin mx-auto mb-4" />
                  <p className="text-text-primary font-medium">Scanning for {selectedDevice?.name}...</p>
                  <p className="text-sm text-text-muted mt-2">
                    Make sure your device is turned on and in pairing mode.
                  </p>
                </>
              ) : discoveredDevices.length > 0 ? (
                <>
                  <p className="text-text-primary font-medium mb-4">Device Found</p>
                  {discoveredDevices.map((dev) => (
                    <button
                      key={dev.id}
                      onClick={() => handleConnectDevice(dev.id, dev.name)}
                      disabled={isConnecting}
                      className="w-full flex items-center gap-3 p-4 bg-bg-elevated rounded-lg hover:bg-bg-highlight"
                    >
                      <Bluetooth className="w-6 h-6 text-primary" />
                      <span className="flex-1 text-left text-text-primary font-medium">{dev.name}</span>
                      {isConnecting ? (
                        <Loader2 className="w-5 h-5 animate-spin text-primary" />
                      ) : (
                        <ChevronRight className="w-5 h-5 text-text-muted" />
                      )}
                    </button>
                  ))}
                </>
              ) : (
                <>
                  <p className="text-text-primary font-medium">No Devices Found</p>
                  <p className="text-sm text-text-muted mt-2 mb-4">
                    Make sure your device is on and in pairing mode.
                  </p>
                  <button
                    onClick={() => selectedDevice && handleSelectDevice(selectedDevice)}
                    className="btn-secondary"
                  >
                    Try Again
                  </button>
                </>
              )}
            </div>
          )}

          {step === 'handy' && (
            <div className="py-4">
              <div className="w-12 h-12 rounded-full bg-primary/20 flex items-center justify-center mx-auto mb-4">
                <Zap className="w-6 h-6 text-primary" />
              </div>
              <p className="text-center text-text-secondary mb-6">
                Enter your Handy connection key. You can find it in the Handy app or at handyfeeling.com
              </p>
              <input
                type="text"
                placeholder="Enter connection key"
                value={handyKey}
                onChange={(e) => setHandyKey(e.target.value.toUpperCase())}
                className="w-full px-4 py-3 bg-bg-elevated border border-border rounded-lg text-text-primary placeholder-text-muted focus:outline-none focus:border-primary mb-4"
              />
              <button
                onClick={handleConnectHandy}
                disabled={!handyKey.trim() || isConnecting}
                className="btn-primary w-full flex items-center justify-center gap-2"
              >
                {isConnecting ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Connecting...
                  </>
                ) : (
                  'Connect'
                )}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default function SettingsPage() {
  const navigate = useNavigate()
  const { user, logout } = useAuthStore()
  const [settings, setSettings] = useState(loadSettings())
  const [connectedDevice, setConnectedDevice] = useState<ConnectedDevice | null>(loadConnectedDevice())
  const [showDeviceModal, setShowDeviceModal] = useState(false)
  const [isSaving, setIsSaving] = useState(false)

  useEffect(() => {
    saveSettings(settings)
  }, [settings])

  const handleThemeChange = (theme: Theme) => {
    setSettings({ ...settings, theme })
    // Apply theme to document
    document.documentElement.classList.remove('light', 'dark')
    if (theme === 'system') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
      document.documentElement.classList.add(prefersDark ? 'dark' : 'light')
    } else {
      document.documentElement.classList.add(theme)
    }
  }

  const handleDeviceConnected = (device: ConnectedDevice) => {
    setConnectedDevice(device)
    saveConnectedDevice(device)
  }

  const handleDisconnectDevice = () => {
    if (confirm(`Disconnect ${connectedDevice?.name}?`)) {
      setConnectedDevice(null)
      saveConnectedDevice(null)
    }
  }

  const handleLogout = () => {
    if (confirm('Are you sure you want to sign out?')) {
      logout()
      navigate('/')
    }
  }

  const handleSaveSettings = async () => {
    setIsSaving(true)
    await new Promise((r) => setTimeout(r, 500))
    saveSettings(settings)
    setIsSaving(false)
  }

  return (
    <div className="flex-1 overflow-auto p-6">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-text-primary">Settings</h1>
          <p className="text-text-secondary">Manage your account and preferences</p>
        </div>

        {/* Account Section */}
        <SettingSection title="Account">
          <SettingRow icon={User} title={user?.display_name || user?.username || 'User'} description={user?.email}>
            <span className="text-xs text-status-success bg-status-success/20 px-2 py-1 rounded">Active</span>
          </SettingRow>
        </SettingSection>

        {/* Device Section */}
        <SettingSection title="Device">
          {connectedDevice ? (
            <>
              <SettingRow
                icon={Bluetooth}
                title={connectedDevice.name}
                description={`${connectedDevice.manufacturer} • ${connectedDevice.features.join(', ')}`}
              >
                <span className="text-xs text-status-success bg-status-success/20 px-2 py-1 rounded">Connected</span>
              </SettingRow>
              {connectedDevice.batteryLevel !== undefined && (
                <SettingRow icon={Zap} title="Battery Level" description={`${connectedDevice.batteryLevel}%`} />
              )}
              <div className="flex gap-2 p-4">
                <button onClick={() => setShowDeviceModal(true)} className="btn-secondary flex-1 text-sm">
                  Change Device
                </button>
                <button onClick={handleDisconnectDevice} className="btn-secondary flex-1 text-sm text-status-error">
                  Disconnect
                </button>
              </div>
            </>
          ) : (
            <SettingRow
              icon={Bluetooth}
              title="Add Device"
              description="Connect Lovense, Kiiroo, We-Vibe, and 200+ more devices"
              onClick={() => setShowDeviceModal(true)}
            >
              <ChevronRight className="w-5 h-5 text-text-muted" />
            </SettingRow>
          )}
        </SettingSection>

        {/* Appearance Section */}
        <SettingSection title="Appearance">
          <div className="p-4">
            <p className="font-medium text-text-primary mb-3">Theme</p>
            <div className="flex gap-2">
              {[
                { value: 'dark', icon: Moon, label: 'Dark' },
                { value: 'light', icon: Sun, label: 'Light' },
                { value: 'system', icon: Monitor, label: 'System' },
              ].map(({ value, icon: Icon, label }) => (
                <button
                  key={value}
                  onClick={() => handleThemeChange(value as Theme)}
                  className={`flex-1 flex items-center justify-center gap-2 p-3 rounded-lg border transition-colors ${
                    settings.theme === value
                      ? 'border-primary bg-primary/10 text-primary'
                      : 'border-border text-text-secondary hover:border-text-muted'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="text-sm font-medium">{label}</span>
                  {settings.theme === value && <Check className="w-4 h-4" />}
                </button>
              ))}
            </div>
          </div>
        </SettingSection>

        {/* Notifications Section */}
        <SettingSection title="Notifications">
          <SettingRow icon={Bell} title="Push Notifications" description="Receive alerts about processing status">
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={settings.notifications}
                onChange={(e) => setSettings({ ...settings, notifications: e.target.checked })}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-bg-elevated peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
            </label>
          </SettingRow>
        </SettingSection>

        {/* Actions */}
        <div className="space-y-3 mt-8">
          <button
            onClick={handleSaveSettings}
            disabled={isSaving}
            className="btn-primary w-full flex items-center justify-center gap-2"
          >
            {isSaving ? <Loader2 className="w-5 h-5 animate-spin" /> : <Save className="w-5 h-5" />}
            {isSaving ? 'Saving...' : 'Save Settings'}
          </button>
          <button
            onClick={handleLogout}
            className="w-full flex items-center justify-center gap-2 p-3 rounded-lg border border-status-error text-status-error hover:bg-status-error/10 transition-colors"
          >
            <LogOut className="w-5 h-5" />
            Sign Out
          </button>
        </div>
      </div>

      {/* Device Selector Modal */}
      <DeviceSelectorModal
        visible={showDeviceModal}
        onClose={() => setShowDeviceModal(false)}
        onDeviceConnected={handleDeviceConnected}
      />
    </div>
  )
}
