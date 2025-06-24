from flask import Flask, request, render_template
import math

app = Flask(__name__)

#DigitalCommunicationCalculator Class
class DigitalCommunicationCalculator:
    def __init__(self, bandwidth_khz, quantizer_bits, source_encoder_rate, channel_encoder_rate, interleaver_bits):
        self.bandwidth_khz = bandwidth_khz
        self.quantizer_bits = quantizer_bits
        self.source_encoder_rate = source_encoder_rate
        self.channel_encoder_rate = channel_encoder_rate
        self.interleaver_bits = interleaver_bits

    def calculate_sampling_frequency(self):
        return 2 * self.bandwidth_khz  # in kHz

    def calculate_quantization_levels(self):
        return 2 ** self.quantizer_bits

    def calculate_bit_rate_after_source_encoder(self, sampling_frequency):
        bit_rate_before_source_encoder = sampling_frequency * self.quantizer_bits
        bit_rate_after_source_encoder = bit_rate_before_source_encoder * self.source_encoder_rate
        return bit_rate_after_source_encoder

    def calculate_bit_rate_after_channel_encoder(self, bit_rate_after_source_encoder):
        bit_rate_after_channel_encoder = bit_rate_after_source_encoder / self.channel_encoder_rate
        return bit_rate_after_channel_encoder

    def calculate_bit_rate_after_interleaver(self, bit_rate_after_channel_encoder):
        return bit_rate_after_channel_encoder  # Bit rate remains unchanged

    def calculate_all(self):
        sampling_frequency = self.calculate_sampling_frequency()
        quantization_levels = self.calculate_quantization_levels()
        bit_rate_after_source_encoder = self.calculate_bit_rate_after_source_encoder(sampling_frequency)
        bit_rate_after_channel_encoder = self.calculate_bit_rate_after_channel_encoder(bit_rate_after_source_encoder)
        bit_rate_after_interleaver = self.calculate_bit_rate_after_interleaver(bit_rate_after_channel_encoder)

        return {
            "Sampling Frequency (kHz)": sampling_frequency,
            "The number of quantization levels (levels)": quantization_levels,
            "The bit rate at the output of the source encoder (kbps)": bit_rate_after_source_encoder,
            "The bit rate at the output of the channel encoder (kbps)": bit_rate_after_channel_encoder,
            "The bit rate at the output of the interleaver (kbps)": bit_rate_after_interleaver
        }

###################### LTESystemCalculator class ####################
class LTESystemCalculator:
    def __init__(self, bandwidth, OFDM_symbols, resource_block_duration, modulation_bits, parallel_resource):
        self.bandwidth = bandwidth
        self.OFDM_symbols = OFDM_symbols
        self.resource_block_duration = resource_block_duration
        self.modulation_bits = modulation_bits
        self.parallel_resource = parallel_resource

    def calculate_resource_element_bits(self):
        return self.modulation_bits  # in kHz

    def calculate_OFDM_symbol_bits(self):
        return (self.bandwidth / 15) * self.modulation_bits

    def calculate_resource_block_bits(self):
        return self.calculate_OFDM_symbol_bits() * self.OFDM_symbols

    def calculate_max_rate(self):
        return (self.calculate_resource_block_bits() * self.parallel_resource) / self.resource_block_duration

    def calculate_all(self):
        resource_element_bits = self.calculate_resource_element_bits()
        OFDM_symbol_bits = self.calculate_OFDM_symbol_bits()
        resource_block_bits = self.calculate_resource_block_bits()
        max_rate = self.calculate_max_rate()

        return {
            "Number of bits per resource element": resource_element_bits,
            "Number of bits per OFDM symbol": OFDM_symbol_bits,
            "Number of bits per OFDM resource block": resource_block_bits,
            "Maximum transmission rate for parallel resource blocks": max_rate
        }

#########################TransmitPowerCalculator class###################

class TransmitPowerCalculator:
    def __init__(self, form_data):
        self.form_data = form_data
        self.path_loss = float(form_data['path_loss'])
        self.frequency = float(form_data['frequency'])
        self.transmit_antenna_gain = float(form_data['transmit_antenna_gain'])
        self.receive_antenna_gain = float(form_data['receive_antenna_gain'])
        self.data_rate = float(form_data['data_rate'])
        self.feed_line_loss = float(form_data['feed_line_loss'])
        self.other_losses = float(form_data['other_losses'])
        self.fade_margin = float(form_data['fade_margin'])
        self.receiver_amplifier_gain = float(form_data['receiver_amplifier_gain'])
        self.transmit_amplifier_gain = float(form_data['transmit_amplifier_gain'])
        self.noise_figure_total = float(form_data['noise_figure_total'])
        self.noise_temperature = float(form_data['noise_temperature'])
        self.link_margin = float(form_data['link_margin'])
        self.modulation_signal = form_data['modulation_signal']
        self.ber = form_data['ber']


    def calculate_total_transmit_power(self):
        try:
            # Constants
            k = 1.38e-23
            T0 = 290  # Reference temperature in Kelvin

            # Convert to dB
            k_db = 10 * math.log10(k)
            noise_temperature_db = 10 * math.log10(self.noise_temperature)
            data_rate_db = 10 * math.log10(self.data_rate * 1e3)

            # Set Eb/N0 for different modulation signal and BERs
            eb_n0_values = {
                '1e-2': {'BPSK/QPSK': 4, '8-PSK': 8, '16-PSK': 12},
                '1e-4': {'BPSK/QPSK': 8.5, '8-PSK': 12, '16-PSK': 16},
                '1e-6': {'BPSK/QPSK': 10.5, '8-PSK': 14, '16-PSK': 18}
            }
            eb_n0 = eb_n0_values.get(self.ber, {}).get(self.modulation_signal, None)

            if eb_n0 is None:
                raise ValueError('Invalid modulation signal or BER selected')

            # Calculate power receiver
            power_receiver_db = (
                self.link_margin +
                k_db +
                noise_temperature_db +
                self.noise_figure_total +
                data_rate_db +
                eb_n0
            )

            # Calculate total transmit power
            total_transmit_power_db = (
                power_receiver_db +
                self.path_loss +
                self.feed_line_loss +
                self.other_losses +
                self.fade_margin -
                self.transmit_antenna_gain -
                self.receive_antenna_gain -
                self.transmit_amplifier_gain -
                self.receiver_amplifier_gain
            )
            # Convert dB to dBm
            total_transmit_power_dbm = total_transmit_power_db + 30

            # Convert dB to Watts
            total_transmit_power_watt = 10 ** (total_transmit_power_db / 10)


            return {
                'dB': total_transmit_power_db,
                'dBm': total_transmit_power_dbm,
                'Watt': total_transmit_power_watt
            }



        except Exception as e:
            return {'error': str(e)}


####################ThroughputCalculator class#########################
class ThroughputCalculator:
    def __init__(self, form_data):
        self.form_data = form_data
        self.bandwidth = float(form_data['bandwidth'])  # in Mbps
        self.frame_size = float(form_data['frame_size'])  # in Kbits
        self.frame_rate = float(form_data['frame_rate'])  # in kilo frames per second
        self.propagation_time = float(form_data['propagation_time'])  # in microseconds

    def calculate_throughput_percentage(self):
        try:
            # Calculate frame transmission time (T)
            T = (self.frame_size * 10 ** 3) / (self.bandwidth * 10 ** 6)  # in seconds

            # Calculate load (G = g * T)
            G = self.frame_rate * 10 ** 3 * T  # Load

            # Calculate alpha
            alpha = (self.propagation_time * 10 ** -6) / T  # Propagation delay

            # Calculate throughput percentages using different protocols
            throughput_pure_aloha = G * math.exp(-2 * G)
            throughput_slotted_aloha = G * math.exp(-1 * G)
            throughput_unslotted_nonpersistent = (G * math.exp(-2 * alpha * T)) / (
                        G * (1 + 2 * alpha) + math.exp(-1 * alpha * G))
            throughput_slotted_nonpersistent = (alpha * G * math.exp(-2 * alpha * T)) / (
                        1 - math.exp(-1 * alpha * G) + alpha)
            throughput_slotted_1_persistent = (G * (1 + alpha - math.exp(-1 * alpha * G)) * math.exp(
                -1 * G * (1 + alpha))) / (((1 + alpha) * (1 - math.exp(-1 * alpha * G))) + alpha * math.exp(
                -1 * G * (1 + alpha)))

            # Convert to percentages
            throughput_pure_aloha_percentage = throughput_pure_aloha * 100
            throughput_slotted_aloha_percentage = throughput_slotted_aloha * 100
            throughput_unslotted_nonpersistent_percentage = throughput_unslotted_nonpersistent * 100
            throughput_slotted_nonpersistent_percentage = throughput_slotted_nonpersistent * 100
            throughput_slotted_1_persistent_percentage = throughput_slotted_1_persistent * 100

            return {
                'pure ALOHA': throughput_pure_aloha_percentage,
                'slotted ALOHA': throughput_slotted_aloha_percentage,
                'unslotted nonpersistent CSMA': throughput_unslotted_nonpersistent_percentage,
                'slotted nonpersistent CSMA': throughput_slotted_nonpersistent_percentage,
                'slotted 1-persistent CSMA': throughput_slotted_1_persistent_percentage
            }

        except Exception as e:
            return {'error': str(e)}

###################CellularSystemCalculator class##################


import math

class CellularSystemCalculator:
    def __init__(self, timeslots, area, number_of_users, calls_per_day, call_duration, call_drop_probability,
                 SIR, reference_distance, reference_distance_power, path_loss_exponent, receiver_sensitivity):
        self.timeslots = timeslots
        self.area = area
        self.number_of_users = number_of_users
        self.calls_per_day = calls_per_day
        self.call_duration = call_duration
        self.call_drop_probability = call_drop_probability
        self.SIR = SIR
        self.reference_distance = reference_distance
        self.reference_distance_power = reference_distance_power
        self.path_loss_exponent = path_loss_exponent
        self.receiver_sensitivity = receiver_sensitivity

    def calculate_max_distance(self):
        max_distance = self.reference_distance / (
            (self.receiver_sensitivity * 10 ** -6) / (10 ** (self.reference_distance_power / 10))
        ) ** (1 / self.path_loss_exponent)
        return max_distance

    def calculate_cell_size(self):
        cell_size = (self.calculate_max_distance()) ** 2 * 2.598076211
        return cell_size

    def calculate_number_of_cells(self):
        number_of_cells = round(self.area / self.calculate_cell_size())
        return number_of_cells

    def calculate_traffic_load_per_user(self):
        AU = self.number_of_users * self.call_duration * self.calls_per_day * (1 / 1440)
        return AU

    def calculate_traffic_per_cell(self):
        return self.calculate_traffic_load_per_user() / self.calculate_number_of_cells()

    def calculate_number_of_cells_per_cluster(self):
        number_of_cells = (((10 ** (self.SIR / 10)) * 6) ** (2 / self.path_loss_exponent)) * (1 / 3)
        number_of_cells = round(number_of_cells)

        def is_valid_cluster_size(N):
            for i in range(0, int(math.sqrt(N)) + 1):
                for j in range(i, int(math.sqrt(N)) + 1):
                    if i ** 2 + i * j + j ** 2 == N:
                        return True
            return False

        while not is_valid_cluster_size(number_of_cells):
            number_of_cells += 1

        return number_of_cells

    @staticmethod
    def erlang_b(E, m):
        InvB = 1.0
        for j in range(1, m + 1):
            InvB = 1.0 + InvB * (j / E)
        return (1.0 / InvB)

    def findN_B(self, B, E):
        n = 1
        while True:
            B_found = self.erlang_b(E, n)
            if B_found <= B:
                return n
            else:
                n += 1

    def calculate_number_of_channels(self):
        number_of_channels = self.findN_B(self.call_drop_probability, self.calculate_traffic_per_cell())
        return number_of_channels

    def calculate_carriers_per_cell(self):
        number_of_carriers = self.calculate_number_of_channels() / self.timeslots
        number_of_carriers = round(number_of_carriers)
        return number_of_carriers

    def calculate_minimum_number_of_carriers(self):
        return self.calculate_number_of_cells_per_cluster() * self.calculate_carriers_per_cell()

    def calculate_all(self):
        max_distance = self.calculate_max_distance()
        cell_size = self.calculate_cell_size()
        number_of_cells = self.calculate_number_of_cells()
        traffic_load_per_user = self.calculate_traffic_load_per_user()
        traffic_per_cell = self.calculate_traffic_per_cell()
        number_of_cells_per_cluster = self.calculate_number_of_cells_per_cluster()
        number_of_channels = self.calculate_number_of_channels()
        carriers_per_cell = self.calculate_carriers_per_cell()
        minimum_number_of_carriers = self.calculate_minimum_number_of_carriers()

        return {
            "Maximum distance between transmitter and receiver for reliable communication (meter)": max_distance,
            "Maximum cell size assuming hexagonal cells": cell_size,
            "The number of cells in the service area": number_of_cells,
            "Traffic load in the whole cellular system (Erlangs)": traffic_load_per_user,
            "Traffic load in each cell (Erlangs)": traffic_per_cell,
            "Number of cells in each cluster": number_of_cells_per_cluster,
            "Number of channels": number_of_channels,
            "Minimum number of carriers needed per cell": carriers_per_cell,
            "Minimum number of carriers needed (in the whole system)": minimum_number_of_carriers
        }

################Calculators route ##################

@app.route('/digital_communication_calculator', methods=['GET', 'POST'])
def digital_communication_calculator():
    results = None
    form_data = {
        'bandwidth_khz': '',
        'quantizer_bits': '',
        'source_encoder_rate': '',
        'channel_encoder_rate': '',
        'interleaver_bits': ''
    }

    if request.method == 'POST':
        try:
            # Retrieve form data
            bandwidth_khz = float(request.form['bandwidth_khz'])
            quantizer_bits = int(request.form['quantizer_bits'])
            source_encoder_rate = float(request.form['source_encoder_rate'])
            channel_encoder_rate = float(request.form['channel_encoder_rate'])
            interleaver_bits = int(request.form['interleaver_bits'])

            # Store form data for redisplay after calculation
            form_data = {
                'bandwidth_khz': bandwidth_khz,
                'quantizer_bits': quantizer_bits,
                'source_encoder_rate': source_encoder_rate,
                'channel_encoder_rate': channel_encoder_rate,
                'interleaver_bits': interleaver_bits
            }

            # Calculate results
            signal_chain = DigitalCommunicationCalculator(bandwidth_khz, quantizer_bits, source_encoder_rate, channel_encoder_rate, interleaver_bits)
            results = signal_chain.calculate_all()

        except Exception as e:
            results = {'error': str(e)}

    return render_template('index.html', results=results, form_data=form_data)
######################
@app.route('/LTE_system_calculator', methods=['GET', 'POST'])
def LTE_system_calculator():
    results = None
    form_data = {
        'bandwidth': '',
        'OFDM_symbols': '',
        'resource_block_duration': '',
        'modulation_bits': '',
        'parallel_resource': ''
    }
    error_message = None

    if request.method == 'POST':
        try:
            # Retrieve form data
            form_data = {
                'bandwidth': float(request.form['bandwidth']),
                'OFDM_symbols': int(request.form['OFDM_symbols']),
                'resource_block_duration': float(request.form['resource_block_duration']),
                'modulation_bits': int(request.form['modulation_bits']),
                'parallel_resource': int(request.form['parallel_resource'])
            }

            # Check if modulation_bits is divisible by 15
            if form_data['bandwidth'] % 15 != 0:
                raise ValueError("Bandwidth must be divisible by 15 without a remainder.")

            # Calculate results using LTESystemCalculator
            lte_calculator = LTESystemCalculator(
                form_data['bandwidth'],
                form_data['OFDM_symbols'],
                form_data['resource_block_duration'],
                form_data['modulation_bits'],
                form_data['parallel_resource']
            )
            results = lte_calculator.calculate_all()
        except Exception as e:
            error_message = str(e)

    return render_template('LTE.html', results=results, form_data=form_data, error_message=error_message)
####################
@app.route('/transmit_power_calculator', methods=['GET', 'POST'])
def transmit_power_calculator():
    results = None
    if request.method == 'POST':
        form_data = request.form
        calculator = TransmitPowerCalculator(form_data)
        results = calculator.calculate_total_transmit_power()

        unit = form_data.get('unit')
        if results and 'error' not in results:
            results = {unit: results[unit]}


    return render_template('index1.html', results=results)

####################
@app.route('/throughput_calculator', methods=['GET', 'POST'])
def throughput_calculator():
    results = None
    form_data = {
        'bandwidth': '',
        'frame_size': '',
        'frame_rate': '',
        'propagation_time': '',
        'mac_protocol': ''}

    if request.method == 'POST':
        form_data = request.form
        calculator = ThroughputCalculator(form_data)
        all_results = calculator.calculate_throughput_percentage()

        mac_protocol = form_data.get('mac_protocol')
        if all_results and 'error' not in all_results:
            results = {mac_protocol: all_results[mac_protocol]}
        else:
            results = all_results

    return render_template('index2.html', form_data=form_data, results=results)

##############################
@app.route('/cellular_system_calculator', methods=['GET', 'POST'])
def cellular_system_calculator():
    results = None
    form_data = {
        'timeslots': '',
        'area': '',
        'number_of_users': '',
        'calls_per_day': '',
        'call_duration': '',
        'call_drop_probability': '',
        'SIR': '',
        'reference_distance': '',
        'reference_distance_power': '',
        'path_loss_exponent': '',
        'receiver_sensitivity': ''
    }

    if request.method == 'POST':
        try:
            # Retrieve form data
            form_data = {
                'timeslots': int(request.form['timeslots']),
                'area': float(request.form['area']),
                'number_of_users': int(request.form['number_of_users']),
                'calls_per_day': int(request.form['calls_per_day']),
                'call_duration': float(request.form['call_duration']),
                'call_drop_probability': float(request.form['call_drop_probability']),
                'SIR': float(request.form['SIR']),
                'reference_distance': float(request.form['reference_distance']),
                'reference_distance_power': float(request.form['reference_distance_power']),
                'path_loss_exponent': float(request.form['path_loss_exponent']),
                'receiver_sensitivity': float(request.form['receiver_sensitivity'])
            }

            # Calculate results using CellularSystemCalculator
            calculator = CellularSystemCalculator(
                form_data['timeslots'],
                form_data['area'],
                form_data['number_of_users'],
                form_data['calls_per_day'],
                form_data['call_duration'],
                form_data['call_drop_probability'],
                form_data['SIR'],
                form_data['reference_distance'],
                form_data['reference_distance_power'],
                form_data['path_loss_exponent'],
                form_data['receiver_sensitivity']
            )

            results = calculator.calculate_all()
        except Exception as e:
            results = {'error': str(e)}

    return render_template('cellular.html', results=results, form_data=form_data)
###############


@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)

