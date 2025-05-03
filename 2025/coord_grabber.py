import csv
import os
from matplotlib import pyplot as plt
import utm


class CoordGrabber():
    """
    A class to read GPS log files, convert GPS coordinates to UTM coordinates,
    """

    def __init__(self, csv_log_file_path) -> None:
        """
        Initializes the coordgrabber with the path to the CSV log file.
        
        :param csv_log_file_path: Path to the CSV log file.
        """

        self.csv_log_file_path = csv_log_file_path
        self.gps_data = []
        self.utm_data = []
        self.normalized_coord = []


    def read_gps_log(self) -> list:
        """
        Reads a GPS log file and returns a list of tuples containing latitude, longitude, and altitude.

        :return: List of tuples with (timestamp, latitude, longitude, altitude).
        """

        self.gps_data = []

        if not os.path.exists(self.csv_log_file_path):
            raise FileNotFoundError(f'File {self.csv_log_file_path} does not exist.')

        with open(self.csv_log_file_path, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)

            if 'CUSTOM.updateTime' not in header or \
            'OSD.latitude' not in header or \
            'OSD.longitude' not in header or \
            'OSD.altitude [m]' not in header:
                raise ValueError("CSV file must contain 'CUSTOM.updateTime', 'OSD.latitude', 'OSD.longitude' and 'OSD.altitude [m]' columns.")

            for row in reader:
                try:
                    t = row[header.index('CUSTOM.updateTime')]
                    lat = float(row[header.index('OSD.latitude')])
                    lon = float(row[header.index('OSD.longitude')])
                    alt = float(row[header.index('OSD.altitude [m]')])
                    self.gps_data.append((t, lat, lon, alt))
                except ValueError as e:
                    print(f"Skipping row due to error: {e}")

        if not self.gps_data:
            raise ValueError('No valid GPS data found in the file.')

        return self.gps_data


    def convert_gps_to_utm(self) -> list:
        """
        Converts GPS coordinates (latitude, longitude) to UTM coordinates (easting, northing).

        :return: List of tuples with (timestamp, hemisphere, zone, zlet, easting, northing, altitude).
        """

        self.utm_data = []
        utm_converter = utm.utmconv()

        if not self.gps_data:
            raise ValueError('No GPS data to convert.')

        for t, lat, lon, alt in self.gps_data:
            hemisphere, zone, zlet, easting, northing = utm_converter.geodetic_to_utm(lat, lon)
            self.utm_data.append((t, hemisphere, zone, zlet, easting, northing, alt))

        return self.utm_data


    def normalize_coordinates(self) -> list:
        """
        Normalizes UTM coordinates (easting, northing) to a common origin.

        :return: List of tuples with normalized coordinates.
        """

        if not self.utm_data:
            raise ValueError("No UTM data to normalize.")

        # Extract the first easting and northing for normalization
        first_hemisphere = self.utm_data[0][1]
        first_zone = self.utm_data[0][2]
        first_zlet = self.utm_data[0][3]
        first_easting = self.utm_data[0][4]
        first_northing = self.utm_data[0][5]
        first_altitude = self.utm_data[0][6]

        # Check for consistent UTM zone and hemisphere
        for data in self.utm_data:
            if data[1] != first_hemisphere or data[2] != first_zone or data[3] != first_zlet:
                raise ValueError('Inconsistent UTM zone or hemisphere in the data.')

        # Normalize the coordinates
        self.normalized_coord = []
        for data in self.utm_data:
            t, hemisphere, zone, zlet, easting, northing, alt = data
            self.normalized_coord.append((easting - first_easting,
                                          northing - first_northing,
                                          alt - first_altitude))

        return self.normalized_coord


    def plot_utm_data(self) -> None:
        """
        Plots UTM coordinates (easting, northing) from the GPS data.
        """

        if not self.utm_data:
            raise ValueError('No UTM data to plot.')

        timestamps = [data[0] for data in self.utm_data]
        eastings = [data[4] for data in self.utm_data]
        northings = [data[5] for data in self.utm_data]
        altitudes = [data[6] for data in self.utm_data]

        # Normalize coordinates
        normalized_data = self.normalize_coordinates()

        x_coord = [data[0] for data in normalized_data]
        y_coord = [data[1] for data in normalized_data]
        z_coord = [data[2] for data in normalized_data]

        # 3D plot
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # ax.scatter(eastings, northings, altitudes)
        ax.scatter(x_coord, y_coord, z_coord)
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.set_zlabel('Altitude (m)')
        ax.set_title("Drone's path")
        plt.show()


def main() -> None:
    """
    Main function to read GPS log, convert to UTM, and plot the data.
    """

    gps_log_path = 'DJIFlightRecord_2021-03-18_[13-04-51]-TxtLogToCsv.csv'

    cg = CoordGrabber(gps_log_path)

    # Read GPS log
    gps_data = cg.read_gps_log()

    # Convert GPS to UTM
    utm_data = cg.convert_gps_to_utm()

    # Plot UTM data
    cg.plot_utm_data()


# Entry point of the script
if __name__ == "__main__":
    main()
