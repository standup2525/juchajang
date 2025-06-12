// Update parking status every 5 seconds
function updateParkingStatus() {
    fetch('/api/parking-status')
        .then(response => response.json())
        .then(data => {
            // Update parking spots
            document.querySelector('.card-title').textContent = 
                `전체 주차 공간: ${data.total_spots}`;
            document.querySelectorAll('.card-text')[0].textContent = 
                `사용 중: ${data.occupied_spots}`;
            document.querySelectorAll('.card-text')[1].textContent = 
                `남은 공간: ${data.available_spots}`;

            // Update vehicle table
            const tbody = document.querySelector('tbody');
            tbody.innerHTML = '';
            data.vehicles.forEach(vehicle => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${vehicle.plate}</td>
                    <td>${vehicle.entry_time}</td>
                    <td>${vehicle.spot}</td>
                `;
                tbody.appendChild(row);
            });
        })
        .catch(error => console.error('Error:', error));
}

// Initial update
updateParkingStatus();

// Set up periodic updates
setInterval(updateParkingStatus, 5000); 