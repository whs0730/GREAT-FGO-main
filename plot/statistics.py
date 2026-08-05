import numpy as np


def calculate_metrics(errors):
    """Calculate MAE, RMSE and max error."""
    return (np.mean(np.abs(errors)),
            np.sqrt(np.mean(errors ** 2)),
            np.max(np.abs(errors)))


def calculate_all_metrics(errors_x, errors_y, errors_z, data_type="Position"):
    """Calculate metrics for three directions."""
    metrics = {}
    axes_data = {'x': errors_x, 'y': errors_y, 'z': errors_z}

    # Define labels and units
    configs = {
        "Position": {'labels': ['East', 'North', 'Up'], 'units': 'm'},
        "Velocity": {'labels': ['Vx', 'Vy', 'Vz'], 'units': 'm/s'},
        "Attitude": {'labels': ['Yaw', 'Pitch', 'Roll'], 'units': '°'}
    }
    cfg = configs.get(data_type, {'labels': ['X', 'Y', 'Z'], 'units': ''})

    for i, axis in enumerate(['x', 'y', 'z']):
        mae, rmse, max_err = calculate_metrics(axes_data[axis])
        metrics[axis] = {
            'mae': mae, 'rmse': rmse, 'max_error': max_err,
            'label': cfg['labels'][i], 'unit': cfg['units']
        }

    return metrics


def save_comprehensive_statistics(pos_errors_x, pos_errors_y, pos_errors_z,
                                  vel_errors_x=None, vel_errors_y=None, vel_errors_z=None,
                                  att_errors_x=None, att_errors_y=None, att_errors_z=None,
                                  file_data=None, file_ref=None, output_path=None):
    """Save simplified statistics to file."""

    # Calculate metrics
    pos_metrics = calculate_all_metrics(pos_errors_x, pos_errors_y, pos_errors_z, "Position")
    has_velocity = vel_errors_x is not None
    has_attitude = att_errors_x is not None

    if has_velocity:
        vel_metrics = calculate_all_metrics(vel_errors_x, vel_errors_y, vel_errors_z, "Velocity")
    if has_attitude:
        att_metrics = calculate_all_metrics(att_errors_x, att_errors_y, att_errors_z, "Attitude")

    # Calculate 2D/3D accuracy
    horizontal_errors = np.sqrt(pos_errors_x ** 2 + pos_errors_y ** 2)
    threeD_errors = np.sqrt(pos_errors_x ** 2 + pos_errors_y ** 2 + pos_errors_z ** 2)

    horizontal_mae, horizontal_rmse, horizontal_max = calculate_metrics(horizontal_errors)
    threeD_mae, threeD_rmse, threeD_max = calculate_metrics(threeD_errors)

    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("ERROR STATISTICS SUMMARY\n")
        f.write("=" * 40 + "\n")
        if file_data: f.write(f"Data: {file_data}\n")
        if file_ref: f.write(f"Reference: {file_ref}\n")
        f.write(f"Epochs: {len(pos_errors_x)}\n")
        f.write("=" * 40 + "\n\n")

        # Position summary
        f.write("POSITION ERRORS (m)\n")
        f.write("-" * 25 + "\n")
        for axis in ['x', 'y', 'z']:
            m = pos_metrics[axis]
            f.write(f"{m['label']:6} MAE:{m['mae']:6.3f} RMSE:{m['rmse']:6.3f} Max:{m['max_error']:6.3f}\n")

        f.write(f"\nHorizontal - MAE:{horizontal_mae:6.3f} RMSE:{horizontal_rmse:6.3f} Max:{horizontal_max:6.3f}")
        f.write(f"\n3D        - MAE:{threeD_mae:6.3f} RMSE:{threeD_rmse:6.3f} Max:{threeD_max:6.3f}\n")

        # Velocity summary
        if has_velocity:
            f.write("\nVELOCITY ERRORS (m/s)\n")
            f.write("-" * 25 + "\n")
            for axis in ['x', 'y', 'z']:
                m = vel_metrics[axis]
                f.write(f"{m['label']:6} MAE:{m['mae']:6.3f} RMSE:{m['rmse']:6.3f} Max:{m['max_error']:6.3f}\n")

        # Attitude summary
        if has_attitude:
            f.write("\nATTITUDE ERRORS (°)\n")
            f.write("-" * 25 + "\n")
            for axis in ['x', 'y', 'z']:
                m = att_metrics[axis]
                f.write(f"{m['label']:6} MAE:{m['mae']:6.3f} RMSE:{m['rmse']:6.3f} Max:{m['max_error']:6.3f}\n")


def print_statistics(metrics, data_type="Position"):
    """Print statistics to console."""
    print(f"\n{data_type} Errors:")
    for axis in ['x', 'y', 'z']:
        m = metrics[axis]
        print(f"{m['label']:6} MAE:{m['mae']:6.3f} RMSE:{m['rmse']:6.3f} Max:{m['max_error']:6.3f}{m['unit']}")