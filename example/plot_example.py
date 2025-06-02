if __name__ == '__main__':
    from lab_driver import get_path_to_project, get_repo_name
    from lab_driver.process_plots import plot_transfer_function_norm, plot_transfer_function_metric
    from lab_driver.process_data import ProcessTransferFunction

    print(get_repo_name())
    path2data = get_path_to_project(new_folder='test_data', folder_ref=get_repo_name())
    path2runs = get_path_to_project(new_folder='runs', folder_ref=get_repo_name())

    hndl = ProcessTransferFunction()
    ovr = hndl.get_data_overview(
        path=path2data,
        acronym='dac'
    )
    trns = hndl.process_data(
        path=path2data,
        filename=ovr[0]
    )

    plot_transfer_function_norm(
        data=trns,
        path2save=path2runs
    )
    plot_transfer_function_metric(
        data=trns,
        func=hndl.calculate_dnl,
        path2save=path2runs
    )
