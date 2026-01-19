// Comprehensive device database from IoST Index
// All devices compatible with Buttplug.io and FunGen

export interface DeviceModel {
  name: string;
  bluetoothNames: string[]; // Names the device might advertise as
  features: ('vibrate' | 'rotate' | 'linear' | 'oscillate' | 'constrict' | 'inflate' | 'position')[];
  connectivity: ('BT4LE' | 'WiFi' | 'USB' | 'Serial')[];
  notes?: string;
}

export interface Manufacturer {
  name: string;
  displayName: string;
  devices: DeviceModel[];
}

export const deviceDatabase: Manufacturer[] = [
  {
    name: 'lovense',
    displayName: 'Lovense',
    devices: [
      { name: 'Lush', bluetoothNames: ['LVS-Lush', 'LVS-S', 'LVS-Z'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Lush 2', bluetoothNames: ['LVS-Lush2', 'LVS-S2'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Lush 3', bluetoothNames: ['LVS-Lush3', 'LVS-S3'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Max', bluetoothNames: ['LVS-Max', 'LVS-B'], features: ['vibrate', 'constrict'], connectivity: ['BT4LE'] },
      { name: 'Max 2', bluetoothNames: ['LVS-Max2', 'LVS-B2'], features: ['vibrate', 'constrict'], connectivity: ['BT4LE'] },
      { name: 'Nora', bluetoothNames: ['LVS-Nora', 'LVS-A'], features: ['vibrate', 'rotate'], connectivity: ['BT4LE'] },
      { name: 'Hush', bluetoothNames: ['LVS-Hush', 'LVS-P'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Hush 2', bluetoothNames: ['LVS-Hush2', 'LVS-P2'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Edge', bluetoothNames: ['LVS-Edge', 'LVS-C'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Edge 2', bluetoothNames: ['LVS-Edge2', 'LVS-C2'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Domi', bluetoothNames: ['LVS-Domi', 'LVS-W'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Domi 2', bluetoothNames: ['LVS-Domi2', 'LVS-W2'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Osci', bluetoothNames: ['LVS-Osci', 'LVS-O'], features: ['oscillate'], connectivity: ['BT4LE'] },
      { name: 'Osci 2', bluetoothNames: ['LVS-Osci2', 'LVS-O2'], features: ['oscillate'], connectivity: ['BT4LE'] },
      { name: 'Ambi', bluetoothNames: ['LVS-Ambi', 'LVS-L'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Diamo', bluetoothNames: ['LVS-Diamo', 'LVS-J'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Dolce', bluetoothNames: ['LVS-Dolce', 'LVS-Quake'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Ferri', bluetoothNames: ['LVS-Ferri', 'LVS-F'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Exomoon', bluetoothNames: ['LVS-Exomoon'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Flexer', bluetoothNames: ['LVS-Flexer'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Gemini', bluetoothNames: ['LVS-Gemini'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Gush', bluetoothNames: ['LVS-Gush'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Hyphy', bluetoothNames: ['LVS-Hyphy'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Mission', bluetoothNames: ['LVS-Mission'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Calor', bluetoothNames: ['LVS-Calor'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Gravity', bluetoothNames: ['LVS-Gravity'], features: ['vibrate', 'linear'], connectivity: ['BT4LE'] },
      { name: 'Ridge', bluetoothNames: ['LVS-Ridge'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Sex Machine', bluetoothNames: ['LVS-SexMachine'], features: ['linear'], connectivity: ['BT4LE'] },
      { name: 'Solace', bluetoothNames: ['LVS-Solace'], features: ['linear'], connectivity: ['BT4LE'] },
      { name: 'Solace Pro', bluetoothNames: ['LVS-SolacePro'], features: ['linear'], connectivity: ['BT4LE'] },
      { name: 'Tenera', bluetoothNames: ['LVS-Tenera'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Tenera 2', bluetoothNames: ['LVS-Tenera2'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Vulse', bluetoothNames: ['LVS-Vulse'], features: ['vibrate', 'linear'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'kiiroo',
    displayName: 'Kiiroo',
    devices: [
      { name: 'KEON', bluetoothNames: ['KEON', 'Keon'], features: ['linear'], connectivity: ['BT4LE'] },
      { name: 'Onyx', bluetoothNames: ['Onyx'], features: ['linear'], connectivity: ['BT4LE'] },
      { name: 'Onyx+', bluetoothNames: ['Onyx+', 'OnyxPlus'], features: ['linear'], connectivity: ['BT4LE'] },
      { name: 'Onyx 2', bluetoothNames: ['Onyx2'], features: ['linear'], connectivity: ['BT4LE'] },
      { name: 'Pearl', bluetoothNames: ['Pearl'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Pearl 2', bluetoothNames: ['Pearl2'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Pearl 3', bluetoothNames: ['Pearl3'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Titan', bluetoothNames: ['Titan'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Cliona', bluetoothNames: ['Cliona'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'OhMiBod Esca', bluetoothNames: ['Esca', 'OhMiBod Esca'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'OhMiBod Esca 2', bluetoothNames: ['Esca2', 'OhMiBod Esca 2'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'OhMiBod Fuse', bluetoothNames: ['Fuse', 'OhMiBod Fuse'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'OhMiBod Lumen', bluetoothNames: ['Lumen', 'OhMiBod Lumen'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'FeelConnect', bluetoothNames: ['FeelConnect'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'PowerBlow', bluetoothNames: ['PowerBlow'], features: ['linear'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'thehandy',
    displayName: 'The Handy',
    devices: [
      { name: 'The Handy', bluetoothNames: ['Handy', 'TheHandy'], features: ['linear'], connectivity: ['WiFi', 'BT4LE'], notes: 'Primary connection via WiFi with connection key' },
    ],
  },
  {
    name: 'wevibe',
    displayName: 'We-Vibe',
    devices: [
      { name: 'Sync', bluetoothNames: ['Sync', 'We-Vibe Sync'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Sync O', bluetoothNames: ['Sync O', 'SyncO'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Melt', bluetoothNames: ['Melt', 'We-Vibe Melt'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Jive', bluetoothNames: ['Jive', 'We-Vibe Jive'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Chorus', bluetoothNames: ['Chorus', 'We-Vibe Chorus'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Nova', bluetoothNames: ['Nova', 'We-Vibe Nova'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Nova 2', bluetoothNames: ['Nova 2', 'We-Vibe Nova 2'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Rave', bluetoothNames: ['Rave', 'We-Vibe Rave'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Rave 2', bluetoothNames: ['Rave 2', 'We-Vibe Rave 2'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Wish', bluetoothNames: ['Wish', 'We-Vibe Wish'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Vector', bluetoothNames: ['Vector', 'We-Vibe Vector'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Verge', bluetoothNames: ['Verge', 'We-Vibe Verge'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Pivot', bluetoothNames: ['Pivot', 'We-Vibe Pivot'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Bond', bluetoothNames: ['Bond', 'We-Vibe Bond'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Moxie', bluetoothNames: ['Moxie', 'We-Vibe Moxie'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Tango', bluetoothNames: ['Tango', 'We-Vibe Tango'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Tango X', bluetoothNames: ['Tango X', 'We-Vibe Tango X'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Touch', bluetoothNames: ['Touch', 'We-Vibe Touch'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Touch X', bluetoothNames: ['Touch X', 'We-Vibe Touch X'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Ditto', bluetoothNames: ['Ditto', 'We-Vibe Ditto'], features: ['vibrate'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'satisfyer',
    displayName: 'Satisfyer',
    devices: [
      { name: 'Curvy 1+', bluetoothNames: ['SF Curvy 1', 'Curvy 1'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Curvy 2+', bluetoothNames: ['SF Curvy 2', 'Curvy 2'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Curvy 3+', bluetoothNames: ['SF Curvy 3', 'Curvy 3'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Dual Pleasure', bluetoothNames: ['SF Dual Pleasure', 'Dual Pleasure'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Double Joy', bluetoothNames: ['SF Double Joy', 'Double Joy'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Double Plus', bluetoothNames: ['SF Double Plus', 'Double Plus'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Elf', bluetoothNames: ['SF Elf', 'Elf'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Endless Love', bluetoothNames: ['SF Endless Love', 'Endless Love'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Heat Wave', bluetoothNames: ['SF Heat Wave', 'Heat Wave'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'High Fly', bluetoothNames: ['SF High Fly', 'High Fly'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Hot Lover', bluetoothNames: ['SF Hot Lover', 'Hot Lover'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Love Triangle', bluetoothNames: ['SF Love Triangle', 'Love Triangle'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Luxury', bluetoothNames: ['SF Luxury', 'Luxury'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Partner Whale', bluetoothNames: ['SF Partner Whale', 'Partner Whale'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Power Ring', bluetoothNames: ['SF Power Ring', 'Power Ring'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Pro', bluetoothNames: ['SF Pro', 'Satisfyer Pro'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Pro G-Spot Rabbit', bluetoothNames: ['SF Pro G-Spot Rabbit'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Royal', bluetoothNames: ['SF Royal', 'Royal'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Sexy Secret', bluetoothNames: ['SF Sexy Secret', 'Sexy Secret'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Sweet Seal', bluetoothNames: ['SF Sweet Seal', 'Sweet Seal'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Threesome', bluetoothNames: ['SF Threesome', 'Threesome'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Top Secret', bluetoothNames: ['SF Top Secret', 'Top Secret'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Yummy Sunshine', bluetoothNames: ['SF Yummy Sunshine', 'Yummy Sunshine'], features: ['vibrate'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'magicmotion',
    displayName: 'Magic Motion',
    devices: [
      { name: 'Flamingo', bluetoothNames: ['Flamingo', 'MM Flamingo'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Solstice', bluetoothNames: ['Solstice', 'MM Solstice'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Smart Mini Vibe', bluetoothNames: ['Smart Mini', 'MM Smart Mini'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Dante', bluetoothNames: ['Dante', 'MM Dante'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Crystal', bluetoothNames: ['Crystal', 'MM Crystal'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Eidolon', bluetoothNames: ['Eidolon', 'MM Eidolon'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Equinox', bluetoothNames: ['Equinox', 'MM Equinox'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Awaken', bluetoothNames: ['Awaken', 'MM Awaken'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Bobi', bluetoothNames: ['Bobi', 'MM Bobi'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Candy', bluetoothNames: ['Candy', 'MM Candy'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Fugu', bluetoothNames: ['Fugu', 'MM Fugu'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Nyx', bluetoothNames: ['Nyx', 'MM Nyx'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Zenith', bluetoothNames: ['Zenith', 'MM Zenith'], features: ['vibrate'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'svakom',
    displayName: 'Svakom',
    devices: [
      { name: 'Alex Neo', bluetoothNames: ['Alex Neo', 'SVAKOM Alex Neo'], features: ['vibrate', 'linear'], connectivity: ['BT4LE'] },
      { name: 'Ella Neo', bluetoothNames: ['Ella Neo', 'SVAKOM Ella Neo'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Sam Neo', bluetoothNames: ['Sam Neo', 'SVAKOM Sam Neo'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Connection', bluetoothNames: ['Connection', 'SVAKOM Connection'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Echo', bluetoothNames: ['Echo', 'SVAKOM Echo'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Emma', bluetoothNames: ['Emma', 'SVAKOM Emma'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Emma Neo', bluetoothNames: ['Emma Neo', 'SVAKOM Emma Neo'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Iris', bluetoothNames: ['Iris', 'SVAKOM Iris'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Phoenix Neo', bluetoothNames: ['Phoenix Neo', 'SVAKOM Phoenix Neo'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Pulse Union', bluetoothNames: ['Pulse Union', 'SVAKOM Pulse Union'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Tyler', bluetoothNames: ['Tyler', 'SVAKOM Tyler'], features: ['vibrate'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'lelo',
    displayName: 'LELO',
    devices: [
      { name: 'F1 SDK', bluetoothNames: ['F1S', 'LELO F1S'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'F1S V2', bluetoothNames: ['F1SV2', 'LELO F1S V2'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Hugo', bluetoothNames: ['Hugo', 'LELO Hugo'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Tiani', bluetoothNames: ['Tiani', 'LELO Tiani'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Tiani 2', bluetoothNames: ['Tiani 2', 'LELO Tiani 2'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Sona', bluetoothNames: ['Sona', 'LELO Sona'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Sona 2', bluetoothNames: ['Sona 2', 'LELO Sona 2'], features: ['vibrate'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'mysteryvibe',
    displayName: 'MysteryVibe',
    devices: [
      { name: 'Crescendo', bluetoothNames: ['Crescendo', 'MV Crescendo'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Tenuto', bluetoothNames: ['Tenuto', 'MV Tenuto'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Tenuto 2', bluetoothNames: ['Tenuto 2', 'MV Tenuto 2'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Poco', bluetoothNames: ['Poco', 'MV Poco'], features: ['vibrate'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'hismith',
    displayName: 'Hismith',
    devices: [
      { name: 'Pro Traveler', bluetoothNames: ['Pro Traveler', 'Hismith Pro Traveler'], features: ['linear'], connectivity: ['BT4LE'] },
      { name: 'Premium', bluetoothNames: ['Premium', 'Hismith Premium'], features: ['linear'], connectivity: ['BT4LE'] },
      { name: 'Table Top', bluetoothNames: ['Table Top', 'Hismith Table Top'], features: ['linear'], connectivity: ['BT4LE'] },
      { name: 'Capsule', bluetoothNames: ['Capsule', 'Hismith Capsule'], features: ['linear'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'vorze',
    displayName: 'Vorze',
    devices: [
      { name: 'A10 Cyclone SA', bluetoothNames: ['CycSA', 'Vorze CycSA', 'A10 Cyclone SA'], features: ['rotate'], connectivity: ['BT4LE'] },
      { name: 'A10 Piston SA', bluetoothNames: ['PisSA', 'Vorze PisSA', 'A10 Piston SA'], features: ['linear'], connectivity: ['BT4LE'] },
      { name: 'Bach', bluetoothNames: ['Bach', 'Vorze Bach'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'UFO TW', bluetoothNames: ['UFO TW', 'Vorze UFO TW'], features: ['rotate'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'youcups',
    displayName: 'Youcups',
    devices: [
      { name: 'Warrior II', bluetoothNames: ['Warrior', 'Youcups Warrior'], features: ['linear'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'motorbunny',
    displayName: 'Motorbunny',
    devices: [
      { name: 'Motorbunny', bluetoothNames: ['Motorbunny', 'MB'], features: ['vibrate', 'rotate'], connectivity: ['BT4LE'] },
      { name: 'Buck', bluetoothNames: ['Buck', 'Motorbunny Buck'], features: ['vibrate', 'rotate'], connectivity: ['BT4LE'] },
      { name: 'Link', bluetoothNames: ['Link', 'Motorbunny Link'], features: ['vibrate'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'jejoue',
    displayName: 'Je Joue',
    devices: [
      { name: 'Dua', bluetoothNames: ['Dua', 'Je Joue Dua'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Fifi', bluetoothNames: ['Fifi', 'Je Joue Fifi'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'G-Spot Bullet', bluetoothNames: ['G-Spot Bullet', 'Je Joue G-Spot Bullet'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Mimi', bluetoothNames: ['Mimi', 'Je Joue Mimi'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Nuo', bluetoothNames: ['Nuo', 'Je Joue Nuo'], features: ['vibrate'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'prettylove',
    displayName: 'Pretty Love',
    devices: [
      { name: 'Augusta', bluetoothNames: ['Augusta', 'Pretty Love Augusta'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Baird II', bluetoothNames: ['Baird', 'Pretty Love Baird'], features: ['vibrate', 'linear'], connectivity: ['BT4LE'] },
      { name: 'Richardson', bluetoothNames: ['Richardson', 'Pretty Love Richardson'], features: ['vibrate'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'adrienlastic',
    displayName: 'Adrien Lastic',
    devices: [
      { name: 'Inspiration', bluetoothNames: ['Inspiration', 'AL Inspiration'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Gladiator', bluetoothNames: ['Gladiator', 'AL Gladiator'], features: ['vibrate'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'cachito',
    displayName: 'Cachito',
    devices: [
      { name: 'Air Touch', bluetoothNames: ['Air Touch', 'Cachito Air Touch'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Bird', bluetoothNames: ['Bird', 'Cachito Bird'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Dodo', bluetoothNames: ['Dodo', 'Cachito Dodo'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Flamingo', bluetoothNames: ['Flamingo', 'Cachito Flamingo'], features: ['vibrate'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'ohmibod',
    displayName: 'OhMiBod',
    devices: [
      { name: 'blueMotion', bluetoothNames: ['blueMotion', 'OhMiBod blueMotion'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Esca 2', bluetoothNames: ['Esca2', 'OhMiBod Esca 2'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Fuse', bluetoothNames: ['Fuse', 'OhMiBod Fuse'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Lumen', bluetoothNames: ['Lumen', 'OhMiBod Lumen'], features: ['vibrate'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'funfactory',
    displayName: 'Fun Factory',
    devices: [
      { name: 'Volta', bluetoothNames: ['Volta', 'Fun Factory Volta'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Manta', bluetoothNames: ['Manta', 'Fun Factory Manta'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Miss Bi', bluetoothNames: ['Miss Bi', 'Fun Factory Miss Bi'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Stronic Surf', bluetoothNames: ['Stronic Surf', 'Fun Factory Stronic Surf'], features: ['linear'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'xiuxiuda',
    displayName: 'XiuXiuDa',
    devices: [
      { name: 'Whale', bluetoothNames: ['Whale', 'XiuXiuDa Whale', 'XXD Whale'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Shark', bluetoothNames: ['Shark', 'XiuXiuDa Shark', 'XXD Shark'], features: ['vibrate'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'cowgirl',
    displayName: 'The Cowgirl',
    devices: [
      { name: 'The Cowgirl', bluetoothNames: ['Cowgirl', 'The Cowgirl'], features: ['vibrate', 'rotate'], connectivity: ['BT4LE', 'WiFi'] },
      { name: 'The Cowgirl Premium', bluetoothNames: ['Cowgirl Premium', 'The Cowgirl Premium'], features: ['vibrate', 'rotate'], connectivity: ['BT4LE', 'WiFi'] },
    ],
  },
  {
    name: 'fleshlight',
    displayName: 'Fleshlight',
    devices: [
      { name: 'Launch', bluetoothNames: ['Fleshlight Launch', 'Launch'], features: ['linear'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'autoblow',
    displayName: 'Autoblow',
    devices: [
      { name: 'Autoblow AI', bluetoothNames: ['Autoblow', 'Autoblow AI'], features: ['linear'], connectivity: ['BT4LE'] },
      { name: 'Autoblow AI+', bluetoothNames: ['Autoblow AI+', 'Autoblow AI Plus'], features: ['linear'], connectivity: ['BT4LE'] },
      { name: 'Ultra', bluetoothNames: ['Autoblow Ultra', 'Ultra'], features: ['linear'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'syncbot',
    displayName: 'Syncbot',
    devices: [
      { name: 'Syncbot', bluetoothNames: ['Syncbot'], features: ['linear', 'rotate'], connectivity: ['BT4LE', 'WiFi'] },
    ],
  },
  {
    name: 'otouch',
    displayName: 'Otouch',
    devices: [
      { name: 'Chiven', bluetoothNames: ['Chiven', 'Otouch Chiven'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Inscup', bluetoothNames: ['Inscup', 'Otouch Inscup'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Pet', bluetoothNames: ['Pet', 'Otouch Pet'], features: ['vibrate'], connectivity: ['BT4LE'] },
      { name: 'Wireless Chiven', bluetoothNames: ['Wireless Chiven', 'Otouch Wireless Chiven'], features: ['vibrate'], connectivity: ['BT4LE'] },
    ],
  },
  {
    name: 'galakuchannel',
    displayName: 'Galaku Channel',
    devices: [
      { name: 'Rocket', bluetoothNames: ['Rocket', 'Galaku Rocket'], features: ['linear'], connectivity: ['BT4LE'] },
      { name: 'Cannon', bluetoothNames: ['Cannon', 'Galaku Cannon'], features: ['linear'], connectivity: ['BT4LE'] },
    ],
  },
];

// Helper functions
export function getAllManufacturers(): { name: string; displayName: string }[] {
  return deviceDatabase.map(m => ({ name: m.name, displayName: m.displayName }));
}

export function getDevicesByManufacturer(manufacturerName: string): DeviceModel[] {
  const manufacturer = deviceDatabase.find(m => m.name === manufacturerName);
  return manufacturer?.devices || [];
}

export function findDeviceByBluetoothName(bluetoothName: string): { manufacturer: Manufacturer; device: DeviceModel } | null {
  for (const manufacturer of deviceDatabase) {
    for (const device of manufacturer.devices) {
      if (device.bluetoothNames.some(name =>
        bluetoothName.toLowerCase().includes(name.toLowerCase()) ||
        name.toLowerCase().includes(bluetoothName.toLowerCase())
      )) {
        return { manufacturer, device };
      }
    }
  }
  return null;
}

export function getAllBluetoothNames(): string[] {
  const names: string[] = [];
  for (const manufacturer of deviceDatabase) {
    for (const device of manufacturer.devices) {
      names.push(...device.bluetoothNames);
    }
  }
  return names;
}

export function getDeviceFeatureDescription(features: DeviceModel['features']): string {
  const descriptions: Record<string, string> = {
    vibrate: 'Vibration',
    rotate: 'Rotation',
    linear: 'Stroking',
    oscillate: 'Oscillation',
    constrict: 'Constriction',
    inflate: 'Inflation',
    position: 'Position Control',
  };
  return features.map(f => descriptions[f] || f).join(', ');
}

export function getConnectivityDescription(connectivity: DeviceModel['connectivity']): string {
  const descriptions: Record<string, string> = {
    BT4LE: 'Bluetooth',
    WiFi: 'WiFi',
    USB: 'USB',
    Serial: 'Serial',
  };
  return connectivity.map(c => descriptions[c] || c).join(', ');
}

// Total device count
export function getTotalDeviceCount(): number {
  return deviceDatabase.reduce((count, m) => count + m.devices.length, 0);
}
